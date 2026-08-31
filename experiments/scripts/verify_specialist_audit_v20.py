"""Independently reconstruct one v20 specialist audit from its paired CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from audit_specialist_diagonal_v20 import (
  AUDIT_AMENDMENT_RELATIVE,
  PAIRED_FIELDS,
  TRANSITION_CLASSES,
  _repair_summary,
)
from diagonal_audit_stats import (
  hierarchical_paired_scene_interval_v19,
  independent_diagonal_scene_gate_v19,
)
from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  FORMAL_AUDIT_SEED,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_D0_EPISODES,
  FORMAL_TARGET_EPISODES,
  PROTOCOL_ID,
  SPECIALIST_MODES,
)

PROTOCOL_RELATIVE = Path("results/online/specialist_v20/protocol.json")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
  return subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout


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


def _delta(rows: list[dict[str, str]], field: str) -> torch.Tensor:
  return torch.tensor(
    [
      int(row[f"final_{field}"]) - int(row[f"baseline_{field}"])
      for row in rows
    ],
    dtype=torch.float64,
  )


def _mean(rows: list[dict[str, str]], field: str) -> float:
  return sum(int(row[field]) for row in rows) / len(rows)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--paired-csv", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  summary_path = args.summary.resolve()
  paired_csv = args.paired_csv.resolve()
  output = args.output.resolve()
  summary = json.loads(summary_path.read_text())
  mode = summary.get("specialist_mode")
  if mode not in SPECIALIST_MODES:
    raise RuntimeError("v20 summary has an undeclared specialist mode")
  protocol_commit = summary.get("protocol_file", {}).get("git_commit", "")
  frozen_protocol = _git_blob(repo, protocol_commit, str(PROTOCOL_RELATIVE))
  protocol_hash = hashlib.sha256(frozen_protocol).hexdigest()
  audit_implementation = summary.get("audit_implementation", {})
  audit_commit = audit_implementation.get("git_commit", "")
  amendment_relative = audit_implementation.get("relative_path", "")
  if Path(amendment_relative) != AUDIT_AMENDMENT_RELATIVE:
    raise RuntimeError("v20 audit amendment path differs")
  amendment_blob = _git_blob(repo, audit_commit, amendment_relative)
  amendment = json.loads(amendment_blob)
  amendment_hash = hashlib.sha256(amendment_blob).hexdigest()
  audit_source_checks = {
    relative: hashlib.sha256(_git_blob(repo, audit_commit, relative)).hexdigest()
    == expected_hash
    for relative, expected_hash in amendment.get(
      "source_file_sha256", {}
    ).items()
  }
  expected_rows = len(FORMAL_ADAPTATION_SEEDS) * (
    FORMAL_TARGET_EPISODES + FORMAL_D0_EPISODES
  )
  conditions = {
    "protocol_id": summary.get("protocol_id") == PROTOCOL_ID,
    "formal_protocol": summary.get("formal_protocol") is True,
    "protocol_hash": summary.get("protocol_file", {}).get("sha256")
    == protocol_hash
    == _sha256(repo / PROTOCOL_RELATIVE),
    "clean_formal_worktree": summary.get("protocol_file", {}).get(
      "tracked_worktree_and_index_clean"
    )
    is True,
    "audit_amendment_hash": audit_implementation.get("sha256")
    == amendment_hash
    == _sha256(repo / AUDIT_AMENDMENT_RELATIVE),
    "audit_amendment_status": amendment.get("status")
    == "prospectively_frozen_before_first_formal_audit_episode_outcome",
    "audit_amendment_boundary": amendment.get(
      "fresh_audit_evidence_boundary", {}
    ).get("formal_audit_episode_outcomes_observed")
    is False,
    "audit_source_hashes": bool(audit_source_checks)
    and all(audit_source_checks.values()),
    "brief_loader_semantics": summary.get(
      "audit_loader_configuration", {}
    ).get("brief_ppo_refinement")
    is True
    and summary.get("audit_loader_configuration", {}).get(
      "legacy_constraint_payload_ignored"
    )
    is True
    and summary.get("audit_loader_configuration", {}).get(
      "actor_or_checkpoint_tensor_modified"
    )
    is False,
    "audit_seed": summary.get("audit_seed") == FORMAL_AUDIT_SEED,
    "bootstrap_seed": summary.get("bootstrap_seed")
    == FORMAL_BOOTSTRAP_SEED,
    "adaptation_seeds": summary.get("adaptation_seeds")
    == list(FORMAL_ADAPTATION_SEEDS),
    "paired_csv_hash": summary.get("paired_episode_metrics", {}).get(
      "sha256"
    )
    == _sha256(paired_csv),
    "paired_csv_rows": summary.get("paired_episode_metrics", {}).get(
      "row_count"
    )
    == expected_rows,
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
    "cbf_independence_gate_absent": summary.get(
      "evaluation_protocol", {}
    ).get("cbf_independence_gate_used")
    is False,
    "joint_claim_absent": summary.get("evaluation_protocol", {}).get(
      "joint_two_specialist_claim_defined"
    )
    is False,
    "five_independent_runs": len(
      summary.get("training_isolation", {}).get("runs", {})
    )
    == 5,
  }
  failed = [name for name, passed in conditions.items() if not passed]
  if failed:
    raise RuntimeError(f"top-level v20 verification failed: {failed}")

  source_checks: dict[str, bool] = {}
  for relative, expected_hash in summary["training_isolation"][
    "source_file_sha256"
  ].items():
    source_checks[relative] = (
      hashlib.sha256(_git_blob(repo, protocol_commit, relative)).hexdigest()
      == expected_hash
    )
  if not all(source_checks.values()):
    raise RuntimeError("a v20 training source differs from the frozen commit")

  with paired_csv.open(newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = reader.fieldnames
  if fieldnames != PAIRED_FIELDS or len(rows) != expected_rows:
    raise RuntimeError("paired v20 CSV schema or row count differs")
  groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
  transition_counts: dict[str, int] = defaultdict(int)
  transition_counts_by_seed: dict[str, dict[str, int]] = {
    str(seed): {name: 0 for name in TRANSITION_CLASSES}
    for seed in FORMAL_ADAPTATION_SEEDS
  }
  for row in rows:
    if row["specialist_mode"] != mode:
      raise RuntimeError("paired CSV contains another specialist mode")
    role = row["evaluation_role"]
    seed = int(row["adaptation_seed"])
    if seed not in FORMAL_ADAPTATION_SEEDS:
      raise RuntimeError("paired CSV contains an undeclared seed")
    if role not in ("target_diagonal_primary", "d0_sanity"):
      raise RuntimeError("paired CSV contains an undeclared role")
    expected_mode = mode if role == "target_diagonal_primary" else "D0"
    if row["evaluation_mode"] != expected_mode:
      raise RuntimeError("paired CSV evaluation mode and role disagree")
    for field in (
      "baseline_success",
      "final_success",
      "baseline_fell",
      "final_fell",
    ):
      if row[field] not in ("0", "1"):
        raise RuntimeError(f"paired CSV {field} is not binary")
    groups[(role, seed)].append(row)
    if role == "target_diagonal_primary":
      transition_counts[row["transition_class"]] += 1
      transition_counts_by_seed[str(seed)][row["transition_class"]] += 1
  expected_groups = {
    (role, seed)
    for role in ("target_diagonal_primary", "d0_sanity")
    for seed in FORMAL_ADAPTATION_SEEDS
  }
  if set(groups) != expected_groups:
    raise RuntimeError("paired CSV does not contain ten declared groups")
  for (role, seed), group in groups.items():
    expected_count = (
      FORMAL_TARGET_EPISODES
      if role == "target_diagonal_primary"
      else FORMAL_D0_EPISODES
    )
    if len(group) != expected_count:
      raise RuntimeError(f"row count differs for {mode}/{role}/{seed}")
    if sorted(int(row["pair_index"]) for row in group) != list(
      range(expected_count)
    ):
      raise RuntimeError(f"pair indices differ for {mode}/{role}/{seed}")
    simulator_pairs = {
      (int(row["evaluation_seed"]), int(row["environment_id"]))
      for row in group
    }
    if len(simulator_pairs) != expected_count:
      raise RuntimeError(
        f"seed/environment pairing is not unique for {mode}/{role}/{seed}"
      )

  target_success_groups = []
  target_fall_groups = []
  d0_success_groups = []
  d0_fall_groups = []
  per_seed_deltas = []
  declared_seed = summary["independent_claim"]["per_adaptation_seed"]
  for seed in FORMAL_ADAPTATION_SEEDS:
    for role, summary_key in (
      ("target_diagonal_primary", "target"),
      ("d0_sanity", "D0"),
    ):
      group = groups[(role, seed)]
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
        _assert_close(
          value,
          declared_seed[str(seed)][summary_key][field],
          f"{mode}/{seed}/{summary_key}/{field}",
        )
      if role == "target_diagonal_primary":
        target_success_groups.append(success_delta)
        target_fall_groups.append(fall_delta)
        per_seed_deltas.append(actual["paired_success_delta"])
      else:
        d0_success_groups.append(success_delta)
        d0_fall_groups.append(fall_delta)
  offset = 0 if mode == "lateral" else 100
  intervals = {
    "target_success": hierarchical_paired_scene_interval_v19(
      target_success_groups,
      bootstrap_samples=10_000,
      bootstrap_seed=FORMAL_BOOTSTRAP_SEED + offset,
    ),
    "target_fall": hierarchical_paired_scene_interval_v19(
      target_fall_groups,
      bootstrap_samples=10_000,
      bootstrap_seed=FORMAL_BOOTSTRAP_SEED + offset + 1,
    ),
    "d0_success": hierarchical_paired_scene_interval_v19(
      d0_success_groups,
      bootstrap_samples=10_000,
      bootstrap_seed=FORMAL_BOOTSTRAP_SEED + offset + 2,
    ),
    "d0_fall": hierarchical_paired_scene_interval_v19(
      d0_fall_groups,
      bootstrap_samples=10_000,
      bootstrap_seed=FORMAL_BOOTSTRAP_SEED + offset + 3,
    ),
  }
  claim = summary["independent_claim"]
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
    raise RuntimeError("reconstructed v20 gate differs")
  if claim["strong_evidence_lcb95_positive"] is not (
    intervals["target_success"][1] > 0.0
  ):
    raise RuntimeError("reconstructed v20 strong-evidence flag differs")
  reconstructed_transitions = _repair_summary(transition_counts)
  reconstructed_transitions["per_adaptation_seed"] = {
    seed: _repair_summary(counts)
    for seed, counts in transition_counts_by_seed.items()
  }
  if reconstructed_transitions != summary["repairs_regressions"]:
    raise RuntimeError("reconstructed repair/regression counts differ")

  verification = {
    "verified": True,
    "verifier": "independent v20 one-specialist paired-CSV reconstruction v1",
    "specialist_mode": mode,
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
    },
    "audit_implementation": {
      "git_commit": audit_commit,
      "amendment_path": amendment_relative,
      "amendment_sha256": amendment_hash,
      "source_checks": audit_source_checks,
    },
    "top_level_conditions": conditions,
    "protocol_commit_source_checks": source_checks,
    "independent_claim": {
      "claim_passed": gate["passed"],
      "gate": gate,
      "strong_evidence_lcb95_positive": intervals["target_success"][1]
      > 0.0,
      "target_success_mean_lcb95_ucb95": intervals["target_success"],
      "target_fall_mean_lcb95_ucb95": intervals["target_fall"],
      "d0_success_mean_lcb95_ucb95": intervals["d0_success"],
      "per_adaptation_seed_success_delta": per_seed_deltas,
    },
    "repairs_regressions": reconstructed_transitions,
    "no_joint_claim": True,
  }
  rendered = json.dumps(verification, indent=2, sort_keys=True) + "\n"
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.tmp")
  temporary.write_text(rendered)
  temporary.replace(output)
  print(rendered, end="")


if __name__ == "__main__":
  main()
