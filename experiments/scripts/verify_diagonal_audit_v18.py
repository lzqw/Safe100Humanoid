"""Independently reconstruct the v18 diagonal audit from its paired CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from diagonal_audit_stats import (
  hierarchical_paired_scene_interval,
  independent_diagonal_scene_gate,
)

MODES = ("lateral", "cbf", "balance")
SEEDS = (42, 142, 242)
EXPECTED_COMMIT = "108e6013d8d1282b095dafcbb14fa16d73fabfe7"
EXPECTED_AUDIT_SEED = 3_100_000
EXPECTED_BOOTSTRAP_SEED = 4_000_000
EXPECTED_TARGET_ROWS = 512
EXPECTED_D0_ROWS = 256
EXPECTED_TOTAL_ROWS = 6_912


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _display_path(path: Path, repo: Path) -> str:
  try:
    return str(path.relative_to(repo))
  except ValueError:
    return str(path)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--paired-csv", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  return parser.parse_args()


def _assert_close(actual: float, expected: float, label: str) -> None:
  if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15):
    raise RuntimeError(f"{label}: actual {actual} != expected {expected}")


def _assert_interval(
  actual: list[float], expected: tuple[float, float, float], label: str
) -> None:
  if len(actual) != 3:
    raise RuntimeError(f"{label}: interval does not contain three values")
  for index, (actual_value, expected_value) in enumerate(
    zip(actual, expected, strict=True)
  ):
    _assert_close(actual_value, expected_value, f"{label}[{index}]")


def _mean(rows: list[dict[str, str]], field: str) -> float:
  return sum(int(row[field]) for row in rows) / len(rows)


def _delta_tensor(rows: list[dict[str, str]], field: str) -> torch.Tensor:
  baseline = f"baseline_{field}"
  final = f"final_{field}"
  return torch.tensor(
    [int(row[final]) - int(row[baseline]) for row in rows],
    dtype=torch.float64,
  )


def _validate_top_level(
  *,
  repo: Path,
  summary: dict[str, Any],
  paired_csv: Path,
) -> dict[str, Any]:
  protocol_path = repo / "results/online/specialist_v18/protocol.json"
  training_manifest_path = (
    repo / "results/online/specialist_v17/formal/training_manifest.json"
  )
  protocol = json.loads(protocol_path.read_text())
  conditions = {
    "protocol_id": summary.get("protocol_id") == "safe100-diagonal-specialist-v18",
    "formal_protocol": summary.get("formal_protocol") is True,
    "protocol_commit": summary.get("protocol_file", {}).get("git_commit")
    == EXPECTED_COMMIT,
    "protocol_hash": summary.get("protocol_file", {}).get("sha256")
    == _sha256(protocol_path),
    "clean_formal_worktree": summary.get("protocol_file", {}).get(
      "tracked_worktree_and_index_clean"
    )
    is True,
    "audit_seed": summary.get("audit_seed") == EXPECTED_AUDIT_SEED,
    "bootstrap_seed": summary.get("bootstrap_seed") == EXPECTED_BOOTSTRAP_SEED,
    "adaptation_seeds": summary.get("adaptation_seeds") == list(SEEDS),
    "paired_csv_hash": summary.get("paired_episode_metrics", {}).get("sha256")
    == _sha256(paired_csv),
    "paired_csv_declared_rows": summary.get("paired_episode_metrics", {}).get(
      "row_count"
    )
    == EXPECTED_TOTAL_ROWS,
    "training_manifest_hash": _sha256(training_manifest_path)
    == protocol["sealed_inputs"]["training_manifest_sha256"],
    "off_diagonal_absent": summary.get("evaluation_protocol", {}).get(
      "off_diagonal_evaluation_performed"
    )
    is False,
    "macro_absent": summary.get("evaluation_protocol", {}).get(
      "macro_average_computed"
    )
    is False,
    "joint_claim_absent": summary.get("joint_conclusion", {}).get("defined")
    is False,
    "all_three_not_required": summary.get("joint_conclusion", {}).get(
      "all_three_required"
    )
    is False,
    "ci_not_a_gate": summary.get("evaluation_protocol", {}).get(
      "individual_confidence_interval_used_as_gate"
    )
    is False,
  }
  failed = [name for name, passed in conditions.items() if not passed]
  if failed:
    raise RuntimeError(f"top-level verification failed: {failed}")
  return {
    "protocol_path": _display_path(protocol_path, repo),
    "protocol_sha256": _sha256(protocol_path),
    "training_manifest_path": _display_path(training_manifest_path, repo),
    "training_manifest_sha256": _sha256(training_manifest_path),
    "conditions": conditions,
  }


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  summary_path = args.summary.resolve()
  paired_csv = args.paired_csv.resolve()
  summary = json.loads(summary_path.read_text())
  top_level = _validate_top_level(
    repo=repo, summary=summary, paired_csv=paired_csv
  )

  with paired_csv.open(newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = reader.fieldnames
  expected_fields = [
    "specialist_mode",
    "evaluation_mode",
    "evaluation_role",
    "adaptation_seed",
    "pair_index",
    "baseline_success",
    "final_success",
    "baseline_fell",
    "final_fell",
    "baseline_failure_type",
    "final_failure_type",
  ]
  if fieldnames != expected_fields:
    raise RuntimeError("paired CSV columns differ from the frozen schema")
  if len(rows) != EXPECTED_TOTAL_ROWS:
    raise RuntimeError(
      f"paired CSV has {len(rows)} rows; expected {EXPECTED_TOTAL_ROWS}"
    )
  groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    mode = row["specialist_mode"]
    role = row["evaluation_role"]
    seed = int(row["adaptation_seed"])
    if mode not in MODES or seed not in SEEDS:
      raise RuntimeError("paired CSV contains an undeclared mode or seed")
    if role not in ("target_diagonal_primary", "d0_sanity"):
      raise RuntimeError("paired CSV contains an off-protocol evaluation role")
    expected_mode = mode if role == "target_diagonal_primary" else "D0"
    if row["evaluation_mode"] != expected_mode:
      raise RuntimeError("paired CSV evaluation mode and role disagree")
    for binary_field in (
      "baseline_success",
      "final_success",
      "baseline_fell",
      "final_fell",
    ):
      if row[binary_field] not in ("0", "1"):
        raise RuntimeError(f"paired CSV {binary_field} is not binary")
    groups[(mode, role, seed)].append(row)

  expected_group_keys = {
    (mode, role, seed)
    for mode in MODES
    for role in ("target_diagonal_primary", "d0_sanity")
    for seed in SEEDS
  }
  if set(groups) != expected_group_keys:
    raise RuntimeError("paired CSV does not contain exactly 18 declared groups")
  for (mode, role, seed), group in groups.items():
    expected_count = (
      EXPECTED_TARGET_ROWS
      if role == "target_diagonal_primary"
      else EXPECTED_D0_ROWS
    )
    if len(group) != expected_count:
      raise RuntimeError(f"row count differs for {mode}/{role}/{seed}")
    pair_indices = sorted(int(row["pair_index"]) for row in group)
    if pair_indices != list(range(expected_count)):
      raise RuntimeError(f"pair indices differ for {mode}/{role}/{seed}")

  reconstructed_claims: dict[str, Any] = {}
  for mode_index, mode in enumerate(MODES):
    target_success_groups: list[torch.Tensor] = []
    target_fall_groups: list[torch.Tensor] = []
    d0_success_groups: list[torch.Tensor] = []
    d0_fall_groups: list[torch.Tensor] = []
    per_seed_deltas: list[float] = []
    for seed in SEEDS:
      declared = summary["independent_claims"][mode]["per_adaptation_seed"][
        str(seed)
      ]
      for role, summary_key in (
        ("target_diagonal_primary", "target"),
        ("d0_sanity", "D0"),
      ):
        group = groups[(mode, role, seed)]
        success_delta = _delta_tensor(group, "success")
        fall_delta = _delta_tensor(group, "fell")
        actual = {
          "baseline_success_rate": _mean(group, "baseline_success"),
          "final_success_rate": _mean(group, "final_success"),
          "paired_success_delta": float(success_delta.mean()),
          "baseline_fall_rate": _mean(group, "baseline_fell"),
          "final_fall_rate": _mean(group, "final_fell"),
          "paired_fall_delta": float(fall_delta.mean()),
        }
        for field, value in actual.items():
          _assert_close(
            value,
            declared[summary_key][field],
            f"{mode}/{seed}/{summary_key}/{field}",
          )
        if role == "target_diagonal_primary":
          target_success_groups.append(success_delta)
          target_fall_groups.append(fall_delta)
          per_seed_deltas.append(actual["paired_success_delta"])
        else:
          d0_success_groups.append(success_delta)
          d0_fall_groups.append(fall_delta)

    intervals = {
      "target_success": hierarchical_paired_scene_interval(
        target_success_groups,
        bootstrap_samples=10000,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED + 100 * mode_index,
      ),
      "target_fall": hierarchical_paired_scene_interval(
        target_fall_groups,
        bootstrap_samples=10000,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED + 100 * mode_index + 1,
      ),
      "d0_success": hierarchical_paired_scene_interval(
        d0_success_groups,
        bootstrap_samples=10000,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED + 100 * mode_index + 2,
      ),
      "d0_fall": hierarchical_paired_scene_interval(
        d0_fall_groups,
        bootstrap_samples=10000,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED + 100 * mode_index + 3,
      ),
    }
    claim = summary["independent_claims"][mode]
    _assert_interval(
      claim["target"]["paired_success_delta_mean_lcb95_ucb95"],
      intervals["target_success"],
      f"{mode}/target_success_interval",
    )
    _assert_interval(
      claim["target"]["paired_fall_delta_mean_lcb95_ucb95"],
      intervals["target_fall"],
      f"{mode}/target_fall_interval",
    )
    _assert_interval(
      claim["D0_sanity"]["paired_success_delta_mean_lcb95_ucb95"],
      intervals["d0_success"],
      f"{mode}/d0_success_interval",
    )
    _assert_interval(
      claim["D0_sanity"]["paired_fall_delta_mean_lcb95_ucb95"],
      intervals["d0_fall"],
      f"{mode}/d0_fall_interval",
    )
    gate = independent_diagonal_scene_gate(
      diagonal_success_delta=intervals["target_success"][0],
      per_seed_success_deltas=per_seed_deltas,
      diagonal_fall_delta=intervals["target_fall"][0],
      d0_success_delta=intervals["d0_success"][0],
    )
    if gate != claim["gate"] or gate["passed"] != claim["claim_passed"]:
      raise RuntimeError(f"reconstructed gate differs for {mode}")
    reconstructed_claims[mode] = {
      "claim_passed": gate["passed"],
      "gate": gate,
      "target_success_mean_lcb95_ucb95": intervals["target_success"],
      "target_fall_mean_lcb95_ucb95": intervals["target_fall"],
      "d0_success_mean_lcb95_ucb95": intervals["d0_success"],
      "per_adaptation_seed_success_delta": per_seed_deltas,
    }

  passed = [mode for mode in MODES if reconstructed_claims[mode]["claim_passed"]]
  failed = [mode for mode in MODES if mode not in passed]
  if passed != summary["passed_specialists"] or failed != summary["failed_specialists"]:
    raise RuntimeError("reconstructed pass/fail lists differ from the summary")

  verification = {
    "verified": True,
    "verifier": "independent paired-CSV reconstruction v1",
    "summary": {
      "path": _display_path(summary_path, repo),
      "sha256": _sha256(summary_path),
    },
    "paired_csv": {
      "path": _display_path(paired_csv, repo),
      "sha256": _sha256(paired_csv),
      "row_count": len(rows),
      "group_count": len(groups),
    },
    "top_level": top_level,
    "independent_claims": reconstructed_claims,
    "passed_specialists": passed,
    "failed_specialists": failed,
    "no_joint_claim": True,
  }
  rendered = json.dumps(verification, indent=2, sort_keys=True) + "\n"
  if args.output is not None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(output)
  print(rendered, end="")


if __name__ == "__main__":
  main()
