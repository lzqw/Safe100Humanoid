"""Independently reconstruct the v19 two-diagonal audit from paired CSV rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch

from diagonal_audit_stats import (
  hierarchical_paired_scene_interval_v19,
  independent_diagonal_scene_gate_v19,
)


MODES = ("lateral", "contact_stability")
SEEDS = (43, 143, 243, 343, 443)
AUDIT_SEED = 5_100_000
BOOTSTRAP_SEED = 6_000_000
TARGET_ROWS = 512
D0_ROWS = 256
TOTAL_ROWS = 7_680
PROTOCOL_RELATIVE = Path("results/online/specialist_v19/protocol.json")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


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
    raise RuntimeError(f"{label} does not contain three values")
  for index, (actual_value, expected_value) in enumerate(
    zip(actual, expected, strict=True)
  ):
    _assert_close(actual_value, expected_value, f"{label}[{index}]")


def _mean(rows: list[dict[str, str]], field: str) -> float:
  return sum(int(row[field]) for row in rows) / len(rows)


def _delta(rows: list[dict[str, str]], field: str) -> torch.Tensor:
  return torch.tensor(
    [int(row[f"final_{field}"]) - int(row[f"baseline_{field}"]) for row in rows],
    dtype=torch.float64,
  )


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
  result = subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  )
  return result.stdout


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  summary_path = args.summary.resolve()
  paired_csv = args.paired_csv.resolve()
  summary = json.loads(summary_path.read_text())
  protocol = json.loads((repo / PROTOCOL_RELATIVE).read_text())
  protocol_commit = summary.get("protocol_file", {}).get("git_commit", "")
  frozen_protocol = _git_blob(repo, protocol_commit, str(PROTOCOL_RELATIVE))
  protocol_hash = _sha256_bytes(frozen_protocol)
  conditions = {
    "protocol_id": summary.get("protocol_id")
    == "safe100-observable-failure-conditioned-v19",
    "formal_protocol": summary.get("formal_protocol") is True,
    "protocol_hash": summary.get("protocol_file", {}).get("sha256")
    == protocol_hash
    == _sha256(repo / PROTOCOL_RELATIVE),
    "clean_formal_worktree": summary.get("protocol_file", {}).get(
      "tracked_worktree_and_index_clean"
    )
    is True,
    "audit_seed": summary.get("audit_seed") == AUDIT_SEED,
    "bootstrap_seed": summary.get("bootstrap_seed") == BOOTSTRAP_SEED,
    "adaptation_seeds": summary.get("adaptation_seeds") == list(SEEDS),
    "paired_csv_hash": summary.get("paired_episode_metrics", {}).get("sha256")
    == _sha256(paired_csv),
    "paired_csv_rows": summary.get("paired_episode_metrics", {}).get("row_count")
    == TOTAL_ROWS,
    "runtime_cbf": summary.get("runtime_cbf") is True,
    "off_diagonal_absent": summary.get("evaluation_protocol", {}).get(
      "off_diagonal_evaluation_performed"
    )
    is False,
    "macro_absent": summary.get("evaluation_protocol", {}).get(
      "macro_average_computed"
    )
    is False,
    "filter_free_absent": summary.get("evaluation_protocol", {}).get(
      "filter_free_evaluation_performed"
    )
    is False,
    "cbf_independence_gate_absent": summary.get("evaluation_protocol", {}).get(
      "cbf_independence_gate_used"
    )
    is False,
    "joint_claim_absent": summary.get("joint_conclusion", {}).get("defined")
    is False,
    "ci_not_gate": summary.get("evaluation_protocol", {}).get(
      "confidence_interval_used_as_gate"
    )
    is False,
    "ten_independent_runs": sum(
      len(records)
      for records in summary.get("training_isolation", {}).get("runs", {}).values()
    )
    == 10,
  }
  failed_conditions = [name for name, passed in conditions.items() if not passed]
  if failed_conditions:
    raise RuntimeError(f"top-level v19 verification failed: {failed_conditions}")

  source_hashes = summary["training_isolation"]["source_file_sha256"]
  source_checks: dict[str, bool] = {}
  for relative, expected_hash in source_hashes.items():
    source_checks[relative] = (
      _sha256_bytes(_git_blob(repo, protocol_commit, relative)) == expected_hash
    )
  if not all(source_checks.values()):
    raise RuntimeError("a training source hash differs from the protocol commit")

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
  if fieldnames != expected_fields or len(rows) != TOTAL_ROWS:
    raise RuntimeError("paired v19 CSV schema or row count differs")
  groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    mode = row["specialist_mode"]
    role = row["evaluation_role"]
    seed = int(row["adaptation_seed"])
    if mode not in MODES or seed not in SEEDS:
      raise RuntimeError("paired CSV contains an undeclared mode or seed")
    if role not in ("target_diagonal_primary", "d0_sanity"):
      raise RuntimeError("paired CSV contains an undeclared evaluation role")
    expected_mode = mode if role == "target_diagonal_primary" else "D0"
    if row["evaluation_mode"] != expected_mode:
      raise RuntimeError("paired CSV mode and role disagree")
    for field in (
      "baseline_success",
      "final_success",
      "baseline_fell",
      "final_fell",
    ):
      if row[field] not in ("0", "1"):
        raise RuntimeError(f"paired CSV {field} is not binary")
    groups[(mode, role, seed)].append(row)
  expected_groups = {
    (mode, role, seed)
    for mode in MODES
    for role in ("target_diagonal_primary", "d0_sanity")
    for seed in SEEDS
  }
  if set(groups) != expected_groups:
    raise RuntimeError("paired CSV does not contain exactly 20 declared groups")
  for (mode, role, seed), group in groups.items():
    expected_count = TARGET_ROWS if role == "target_diagonal_primary" else D0_ROWS
    if len(group) != expected_count:
      raise RuntimeError(f"row count differs for {mode}/{role}/{seed}")
    if sorted(int(row["pair_index"]) for row in group) != list(
      range(expected_count)
    ):
      raise RuntimeError(f"pair indices differ for {mode}/{role}/{seed}")

  reconstructed: dict[str, Any] = {}
  for mode_index, mode in enumerate(MODES):
    target_success_groups: list[torch.Tensor] = []
    target_fall_groups: list[torch.Tensor] = []
    d0_success_groups: list[torch.Tensor] = []
    d0_fall_groups: list[torch.Tensor] = []
    per_seed_deltas: list[float] = []
    for seed in SEEDS:
      declared = summary["independent_claims"][mode]["per_adaptation_seed"][str(seed)]
      for role, summary_key in (
        ("target_diagonal_primary", "target"),
        ("d0_sanity", "D0"),
      ):
        group = groups[(mode, role, seed)]
        success_delta = _delta(group, "success")
        fall_delta = _delta(group, "fell")
        actual = {
          "baseline_success_rate": _mean(group, "baseline_success"),
          "final_success_rate": _mean(group, "final_success"),
          "paired_success_delta": float(success_delta.mean()),
          "baseline_fall_rate": _mean(group, "baseline_fell"),
          "final_fall_rate": _mean(group, "final_fell"),
          "paired_fall_delta": float(fall_delta.mean()),
        }
        for field, value in actual.items():
          _assert_close(value, declared[summary_key][field], f"{mode}/{seed}/{summary_key}/{field}")
        if role == "target_diagonal_primary":
          target_success_groups.append(success_delta)
          target_fall_groups.append(fall_delta)
          per_seed_deltas.append(actual["paired_success_delta"])
        else:
          d0_success_groups.append(success_delta)
          d0_fall_groups.append(fall_delta)
    intervals = {
      "target_success": hierarchical_paired_scene_interval_v19(
        target_success_groups,
        bootstrap_samples=10000,
        bootstrap_seed=BOOTSTRAP_SEED + 100 * mode_index,
      ),
      "target_fall": hierarchical_paired_scene_interval_v19(
        target_fall_groups,
        bootstrap_samples=10000,
        bootstrap_seed=BOOTSTRAP_SEED + 100 * mode_index + 1,
      ),
      "d0_success": hierarchical_paired_scene_interval_v19(
        d0_success_groups,
        bootstrap_samples=10000,
        bootstrap_seed=BOOTSTRAP_SEED + 100 * mode_index + 2,
      ),
      "d0_fall": hierarchical_paired_scene_interval_v19(
        d0_fall_groups,
        bootstrap_samples=10000,
        bootstrap_seed=BOOTSTRAP_SEED + 100 * mode_index + 3,
      ),
    }
    claim = summary["independent_claims"][mode]
    _assert_interval(
      claim["target"]["paired_success_delta_mean_lcb95_ucb95"],
      intervals["target_success"],
      f"{mode}/target_success",
    )
    _assert_interval(
      claim["target"]["paired_fall_delta_mean_lcb95_ucb95"],
      intervals["target_fall"],
      f"{mode}/target_fall",
    )
    _assert_interval(
      claim["D0_sanity"]["paired_success_delta_mean_lcb95_ucb95"],
      intervals["d0_success"],
      f"{mode}/d0_success",
    )
    _assert_interval(
      claim["D0_sanity"]["paired_fall_delta_mean_lcb95_ucb95"],
      intervals["d0_fall"],
      f"{mode}/d0_fall",
    )
    gate = independent_diagonal_scene_gate_v19(
      diagonal_success_delta=intervals["target_success"][0],
      per_seed_success_deltas=per_seed_deltas,
      diagonal_fall_delta=intervals["target_fall"][0],
      d0_success_delta=intervals["d0_success"][0],
    )
    if gate != claim["gate"] or gate["passed"] != claim["claim_passed"]:
      raise RuntimeError(f"reconstructed gate differs for {mode}")
    if claim["strong_evidence_lcb95_positive"] is not (
      intervals["target_success"][1] > 0.0
    ):
      raise RuntimeError(f"reconstructed strong-evidence flag differs for {mode}")
    reconstructed[mode] = {
      "claim_passed": gate["passed"],
      "gate": gate,
      "strong_evidence_lcb95_positive": intervals["target_success"][1] > 0.0,
      "target_success_mean_lcb95_ucb95": intervals["target_success"],
      "target_fall_mean_lcb95_ucb95": intervals["target_fall"],
      "d0_success_mean_lcb95_ucb95": intervals["d0_success"],
      "per_adaptation_seed_success_delta": per_seed_deltas,
    }
  passed = [mode for mode in MODES if reconstructed[mode]["claim_passed"]]
  failed = [mode for mode in MODES if mode not in passed]
  if passed != summary["passed_specialists"] or failed != summary["failed_specialists"]:
    raise RuntimeError("reconstructed v19 pass/fail lists differ")

  verification = {
    "verified": True,
    "verifier": "independent v19 paired-CSV reconstruction v1",
    "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
    "paired_csv": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": len(rows),
      "group_count": len(groups),
    },
    "protocol": {
      "path": str(PROTOCOL_RELATIVE),
      "git_commit": protocol_commit,
      "sha256": protocol_hash,
      "declared_status": protocol["status"],
    },
    "top_level_conditions": conditions,
    "protocol_commit_source_checks": source_checks,
    "independent_claims": reconstructed,
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
