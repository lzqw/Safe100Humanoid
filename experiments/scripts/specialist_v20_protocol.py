"""Shared prospective protocol definitions for specialist v20 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_ID = "safe100-fixed-budget-observable-failure-conditioned-v20"
POLICY_METHOD = "Fixed-Budget Observable Failure-Conditioned Brief PPO v20"
SPECIALIST_MODES = ("lateral", "contact_stability")
FORMAL_ADAPTATION_SEEDS = (73, 173, 273, 373, 473)
FORMAL_AUDIT_SEED = 5_500_000
FORMAL_BOOTSTRAP_SEED = 6_500_000
FORMAL_ROUNDS = 8
FORMAL_TARGET_EPISODES = 512
FORMAL_D0_EPISODES = 256
FORMAL_BOOTSTRAP_SAMPLES = 10_000
CALIBRATION_CANDIDATE_SEEDS = {
  "lateral": tuple(range(8312, 8320)),
  "contact_stability": tuple(range(8217, 8225)),
}
CALIBRATION_EVALUATION_SEED_BASE = {
  "lateral": 4_820_000,
  "contact_stability": 4_810_000,
}


@dataclass(frozen=True)
class FixedBudgetStatus:
  """Scientific validity of one completed v20 adaptation run."""

  actual_rounds: int
  retained_update_count: int
  protocol_valid: bool
  stop_reason: str
  retained_update_count_is_gate: bool = False


def fixed_budget_status(
  *, actual_rounds: int, retained_update_count: int, formal: bool = True
) -> FixedBudgetStatus:
  """Return v20 completion status without using retained updates as a gate."""
  if actual_rounds < 0 or retained_update_count < 0:
    raise ValueError("v20 round and retained-update counts cannot be negative")
  if retained_update_count > actual_rounds:
    raise ValueError("v20 retained updates cannot exceed completed rounds")
  if formal and actual_rounds != FORMAL_ROUNDS:
    raise ValueError(
      f"formal v20 requires exactly {FORMAL_ROUNDS} rounds, got {actual_rounds}"
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


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _iter_seed_values(value: Any) -> Iterable[int]:
  if isinstance(value, bool):
    return
  if isinstance(value, int):
    yield value
  elif isinstance(value, list):
    for item in value:
      if isinstance(item, int) and not isinstance(item, bool):
        yield item


def _historical_seed_records(
  value: Any, *, source: Path, path: str = ""
) -> Iterable[dict[str, Any]]:
  """Yield only declared specialist/calibration randomness, never episode IDs."""
  adaptation_keys = {
    "adaptation_seed",
    "adaptation_seeds",
    "development_adaptation_seed",
    "development_only_adaptation_seeds",
    "revision_2_declared_adaptation_seeds",
    "revision_3_declared_adaptation_seeds",
  }
  audit_keys = {"audit_seed", "bootstrap_seed"}
  calibration_keys = {
    "candidate_seeds",
    "lateral_candidate_seeds",
    "contact_stability_candidate_seeds",
    "evaluation_seed_base",
    "lateral_evaluation_seed_base",
    "contact_stability_evaluation_seed_base",
  }
  if isinstance(value, dict):
    for key, child in value.items():
      child_path = f"{path}.{key}" if path else key
      if key in adaptation_keys:
        kind = "adaptation"
      elif key in audit_keys:
        kind = "audit_or_bootstrap"
      elif key in calibration_keys:
        kind = "calibration"
      else:
        kind = None
      if kind is not None:
        for seed in _iter_seed_values(child):
          yield {
            "kind": kind,
            "seed": seed,
            "source": str(source),
            "json_path": child_path,
          }
      yield from _historical_seed_records(
        child, source=source, path=child_path
      )
  elif isinstance(value, list):
    for index, child in enumerate(value):
      yield from _historical_seed_records(
        child, source=source, path=f"{path}[{index}]"
      )


def historical_randomness_registry(repo: Path) -> list[dict[str, Any]]:
  """Read frozen v17/v18/v19 JSON evidence and return declared randomness."""
  results = repo / "results/online"
  records: list[dict[str, Any]] = []
  for version in ("specialist_v17", "specialist_v18", "specialist_v19"):
    root = results / version
    if not root.is_dir():
      continue
    for path in sorted(root.rglob("*.json")):
      try:
        payload = json.loads(path.read_text())
      except (json.JSONDecodeError, OSError):
        continue
      records.extend(
        _historical_seed_records(
          payload, source=path.relative_to(repo)
        )
      )
  unique = {
    (item["kind"], item["seed"], item["source"], item["json_path"]): item
    for item in records
  }
  return [unique[key] for key in sorted(unique)]


def fresh_randomness_report(
  repo: Path,
  *,
  adaptation_seeds: Iterable[int] = FORMAL_ADAPTATION_SEEDS,
  audit_seed: int = FORMAL_AUDIT_SEED,
  bootstrap_seed: int = FORMAL_BOOTSTRAP_SEED,
  calibration_candidate_seeds: dict[str, Iterable[int]] = (
    CALIBRATION_CANDIDATE_SEEDS
  ),
  calibration_evaluation_seed_bases: dict[str, int] = (
    CALIBRATION_EVALUATION_SEED_BASE
  ),
) -> dict[str, Any]:
  """Prove all proposed v20 randomness is unseen in specialist evidence."""
  adaptation = tuple(adaptation_seeds)
  if adaptation != tuple(sorted(set(adaptation))) or len(adaptation) != 5:
    raise ValueError("v20 requires five ordered, distinct adaptation seeds")
  if set(calibration_candidate_seeds) != set(SPECIALIST_MODES):
    raise ValueError("v20 calibration seed mapping must cover both specialists")
  registry = historical_randomness_registry(repo)
  used = {
    kind: {item["seed"] for item in registry if item["kind"] == kind}
    for kind in ("adaptation", "audit_or_bootstrap", "calibration")
  }
  proposed_calibration = {
    seed
    for values in calibration_candidate_seeds.values()
    for seed in values
  } | set(calibration_evaluation_seed_bases.values())
  collisions = {
    "adaptation": sorted(set(adaptation) & used["adaptation"]),
    "audit_or_bootstrap": sorted(
      {audit_seed, bootstrap_seed} & used["audit_or_bootstrap"]
    ),
    "calibration": sorted(proposed_calibration & used["calibration"]),
  }
  proposed_all = set(adaptation) | {
    audit_seed,
    bootstrap_seed,
  } | proposed_calibration
  internal_collision = len(proposed_all) != (
    len(adaptation) + 2 + len(proposed_calibration)
  )
  passed = not any(collisions.values()) and not internal_collision
  return {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "historical_versions_scanned": [17, 18, 19],
    "proposed": {
      "adaptation_seeds": list(adaptation),
      "audit_seed": audit_seed,
      "bootstrap_seed": bootstrap_seed,
      "calibration_candidate_seeds": {
        mode: list(calibration_candidate_seeds[mode])
        for mode in SPECIALIST_MODES
      },
      "calibration_evaluation_seed_bases": (
        calibration_evaluation_seed_bases
      ),
    },
    "historical_registry": registry,
    "collisions": collisions,
    "proposed_internal_collision": internal_collision,
    "passed": passed,
  }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
  rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite different v20 preflight: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  repo = args.repo.resolve()
  report = fresh_randomness_report(repo)
  if not report["passed"]:
    raise RuntimeError(f"v20 randomness collision: {report['collisions']}")
  if args.output is not None:
    _write_immutable(args.output.resolve(), report)
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
