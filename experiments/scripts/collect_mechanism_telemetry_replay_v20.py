"""Collect first-attempt v20 telemetry replays with explicit disclosure.

The formal audit is authoritative for episode outcomes.  This post-audit
reporting wrapper keeps the prospectively selected lowest pair index, reuses
any already-written first attempt, and records rather than suppresses replay
outcome divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from collect_mechanism_telemetry_v20 import (
  MECHANISM_FIELDS,
  TELEMETRY_FIELDS,
  _read_csv,
  _sha256,
)
from specialist_v20_protocol import FORMAL_ADAPTATION_SEEDS, SPECIALIST_MODES

TRACE_FILENAMES = ("evaluation.json", "episodes.csv", "telemetry.csv")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--telemetry-repo", type=Path, required=True)
  parser.add_argument("--telemetry-commit", required=True)
  parser.add_argument("--telemetry-amendment", type=Path, required=True)
  parser.add_argument("--mode", choices=SPECIALIST_MODES, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--audit-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _git_output(repo: Path, *args: str) -> str:
  completed = subprocess.run(
    ["git", "-C", str(repo), *args],
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _validate_telemetry_boundary(
  *,
  telemetry_repo: Path,
  telemetry_commit: str,
  amendment_path: Path,
  audit_dir: Path,
  mode: str,
) -> dict[str, Any]:
  if _git_output(telemetry_repo, "rev-parse", "HEAD") != telemetry_commit:
    raise RuntimeError("v20 telemetry worktree is not at the frozen commit")
  if _git_output(telemetry_repo, "status", "--porcelain"):
    raise RuntimeError("v20 telemetry worktree is not clean")
  amendment = json.loads(amendment_path.read_text())
  required = {
    "formal_audit_outcomes_are_authoritative": True,
    "formal_audit_or_training_rerun": False,
    "lowest_pair_index_selection_unchanged": True,
    "retry_until_outcome_matches": False,
    "first_attempt_outputs_may_be_reused_but_not_overwritten": True,
  }
  correction = amendment.get("correction", {})
  mismatches = {
    key: {"actual": correction.get(key), "required": value}
    for key, value in required.items()
    if correction.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"v20 telemetry amendment differs: {mismatches}")
  for relative, expected in amendment.get("source_file_sha256", {}).items():
    path = telemetry_repo / relative
    if not path.is_file() or _sha256(path) != expected:
      raise RuntimeError(f"v20 telemetry source hash differs: {relative}")
  recorded = amendment.get("formal_audit_artifacts", {}).get(mode)
  if recorded is not None:
    for name, expected in recorded.items():
      path = audit_dir / name
      if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"v20 recorded formal artifact differs: {name}")
  partial = amendment.get("triggering_failure", {}).get(
    "first_attempt_file_sha256", {}
  )
  if mode == "lateral":
    for relative, expected in partial.items():
      path = audit_dir / relative
      if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(
          f"v20 preserved first-attempt trace differs: {relative}"
        )
  return {
    "path": str(amendment_path),
    "sha256": _sha256(amendment_path),
    "implementation_commit": amendment["implementation_commit"],
    "freeze_commit": telemetry_commit,
    "status": amendment["status"],
  }


def _selected_repairs(
  paired_rows: list[dict[str, str]], mode: str
) -> dict[int, dict[str, str] | None]:
  target_repairs = [
    row
    for row in paired_rows
    if row["specialist_mode"] == mode
    and row["evaluation_role"] == "target_diagonal_primary"
    and row["transition_class"] == "failure_to_success"
  ]
  output: dict[int, dict[str, str] | None] = {}
  for seed in FORMAL_ADAPTATION_SEEDS:
    candidates = [
      row for row in target_repairs if int(row["adaptation_seed"]) == seed
    ]
    output[seed] = (
      min(candidates, key=lambda row: int(row["pair_index"]))
      if candidates
      else None
    )
  return output


def _trace_file_state(trace_dir: Path) -> str:
  present = [(trace_dir / name).is_file() for name in TRACE_FILENAMES]
  if all(present):
    return "complete_first_attempt"
  if any(present):
    raise RuntimeError(
      f"v20 telemetry first attempt is partial and will not be overwritten: "
      f"{trace_dir}"
    )
  return "not_started"


def _formal_evaluation_paths(
  *,
  audit_dir: Path,
  seed: int,
  policy_role: str,
  evaluation_seed: str,
) -> tuple[Path, Path]:
  directory = audit_dir / "raw" / policy_role / f"seed{seed}" / "target"
  json_paths = list(directory.glob(f"*-seed{evaluation_seed}.json"))
  csv_paths = list(directory.glob(f"*-seed{evaluation_seed}.csv"))
  if len(json_paths) != 1 or len(csv_paths) != 1:
    raise RuntimeError("v20 formal evaluation block is not unique")
  return json_paths[0], csv_paths[0]


def _matching_episode(
  rows: list[dict[str, str]], selected: dict[str, str]
) -> dict[str, str]:
  matching = [
    row
    for row in rows
    if row["evaluation_seed"] == selected["evaluation_seed"]
    and row["environment_id"] == selected["environment_id"]
  ]
  if len(matching) != 1:
    raise RuntimeError("v20 telemetry identity did not resolve to one episode")
  return matching[0]


def _binary(value: str) -> int:
  if value in {"1", "True"}:
    return 1
  if value in {"0", "False"}:
    return 0
  raise ValueError(f"v20 telemetry value is not binary: {value}")


def _compare_formal_replay(
  formal: dict[str, str], replay: dict[str, str]
) -> dict[str, bool]:
  return {
    "success": _binary(formal["success"]) == _binary(replay["success"]),
    "fell": _binary(formal["fell"]) == _binary(replay["fell"]),
    "failure_type": formal["failure_type"] == replay["failure_type"],
  }


def _run_or_reuse_trace(
  *,
  repo: Path,
  context: Path,
  audit_dir: Path,
  output_dir: Path,
  seed: int,
  selected: dict[str, str],
  policy_role: str,
  device: str,
) -> dict[str, Any]:
  checkpoint = (
    audit_dir / "raw" / policy_role / f"seed{seed}" / "target" / "actor.pt"
  )
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  trace_dir = output_dir / "raw" / f"seed{seed}" / policy_role
  state = _trace_file_state(trace_dir)
  trace_dir.mkdir(parents=True, exist_ok=True)
  output_json = trace_dir / "evaluation.json"
  output_csv = trace_dir / "episodes.csv"
  telemetry_csv = trace_dir / "telemetry.csv"
  reused = state == "complete_first_attempt"
  if not reused:
    command = [
      sys.executable,
      str(repo / "experiments/scripts/evaluate_online_stairs.py"),
      "--repo",
      str(repo),
      "--task",
      "Unitree-G1-Stairs-Online-DQHMED",
      "--checkpoint",
      str(checkpoint),
      "--num-envs",
      "128",
      "--num-episodes",
      "128",
      "--seed",
      selected["evaluation_seed"],
      "--device",
      device,
      "--runtime-filter",
      "on",
      "--one-episode-per-env",
      "--deployment-context",
      str(context),
      "--v19-context",
      str(context),
      "--telemetry-env-id",
      selected["environment_id"],
      "--telemetry-output-csv",
      str(telemetry_csv),
      "--output-json",
      str(output_json),
      "--output-csv",
      str(output_csv),
    ]
    print(
      f"v20 telemetry first attempt: seed={seed} role={policy_role} "
      f"pair={selected['pair_index']}",
      flush=True,
    )
    completed = subprocess.run(
      command,
      cwd=repo,
      check=False,
      capture_output=True,
      text=True,
    )
    if completed.returncode != 0:
      diagnostic = "\n".join(
        (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
      )
      raise RuntimeError(f"v20 mechanism trace failed:\n{diagnostic}")
  else:
    print(
      f"v20 telemetry reusing preserved first attempt: seed={seed} "
      f"role={policy_role} pair={selected['pair_index']}",
      flush=True,
    )

  replay_summary = json.loads(output_json.read_text())
  replay_fields, replay_rows = _read_csv(output_csv)
  if not replay_fields:
    raise RuntimeError("v20 telemetry replay episode CSV has no schema")
  replay_episode = _matching_episode(replay_rows, selected)
  formal_json, formal_csv = _formal_evaluation_paths(
    audit_dir=audit_dir,
    seed=seed,
    policy_role=policy_role,
    evaluation_seed=selected["evaluation_seed"],
  )
  formal_summary = json.loads(formal_json.read_text())
  _, formal_rows = _read_csv(formal_csv)
  formal_episode = _matching_episode(formal_rows, selected)
  if replay_summary["initial_state_signature"] != formal_summary[
    "initial_state_signature"
  ]:
    raise RuntimeError("v20 telemetry initial-state signature differs")
  if replay_summary["actor_state_sha256"] != formal_summary[
    "actor_state_sha256"
  ]:
    raise RuntimeError("v20 telemetry actor hash differs")
  fields, telemetry_rows = _read_csv(telemetry_csv)
  if fields != TELEMETRY_FIELDS or not telemetry_rows:
    raise RuntimeError("v20 telemetry trace schema or rows differ")
  comparison = _compare_formal_replay(formal_episode, replay_episode)
  return {
    "policy_role": policy_role,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint),
    "actor_state_sha256": replay_summary["actor_state_sha256"],
    "initial_state_signature": replay_summary["initial_state_signature"],
    "initial_state_signature_matches_formal": True,
    "actor_state_sha256_matches_formal": True,
    "first_attempt_reused": reused,
    "retry_until_outcome_matches": False,
    "formal_episode": {
      "success": _binary(formal_episode["success"]),
      "fell": _binary(formal_episode["fell"]),
      "failure_type": formal_episode["failure_type"],
      "evaluation_json": str(formal_json),
      "evaluation_json_sha256": _sha256(formal_json),
      "episode_csv": str(formal_csv),
      "episode_csv_sha256": _sha256(formal_csv),
    },
    "replay_episode": {
      "success": _binary(replay_episode["success"]),
      "fell": _binary(replay_episode["fell"]),
      "failure_type": replay_episode["failure_type"],
    },
    "outcome_field_matches_formal": comparison,
    "outcome_matches_formal": all(comparison.values()),
    "evaluation_json": str(output_json),
    "evaluation_json_sha256": _sha256(output_json),
    "episode_csv": str(output_csv),
    "episode_csv_sha256": _sha256(output_csv),
    "telemetry_csv": str(telemetry_csv),
    "telemetry_csv_sha256": _sha256(telemetry_csv),
    "telemetry_rows": len(telemetry_rows),
  }


def _write_csv(
  path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  telemetry_repo = args.telemetry_repo.resolve()
  context = args.context.resolve()
  audit_dir = args.audit_dir.resolve()
  output_dir = args.output_dir.resolve()
  amendment_path = args.telemetry_amendment.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  amendment = _validate_telemetry_boundary(
    telemetry_repo=telemetry_repo,
    telemetry_commit=args.telemetry_commit,
    amendment_path=amendment_path,
    audit_dir=audit_dir,
    mode=args.mode,
  )
  paired_csv = audit_dir / "paired_episode_metrics.csv"
  _, paired_rows = _read_csv(paired_csv)
  selected_by_seed = _selected_repairs(paired_rows, args.mode)

  combined_rows: list[dict[str, Any]] = []
  selections: dict[str, Any] = {}
  all_traces: list[dict[str, Any]] = []
  for seed, selected in selected_by_seed.items():
    if selected is None:
      selections[str(seed)] = {
        "selected": False,
        "reason": "formal target audit contained no failure_to_success pair",
      }
      continue
    traces = [
      _run_or_reuse_trace(
        repo=repo,
        context=context,
        audit_dir=audit_dir,
        output_dir=output_dir,
        seed=seed,
        selected=selected,
        policy_role=role,
        device=args.device,
      )
      for role in ("baseline", "final")
    ]
    if traces[0]["initial_state_signature"] != traces[1][
      "initial_state_signature"
    ]:
      raise RuntimeError("v20 telemetry baseline/final signatures differ")
    for trace in traces:
      fields, rows = _read_csv(Path(trace["telemetry_csv"]))
      if fields != TELEMETRY_FIELDS:
        raise RuntimeError("v20 mechanism telemetry schemas differ")
      for row in rows:
        combined_rows.append(
          {
            "specialist": args.mode,
            "adaptation_seed": seed,
            "pair_index": selected["pair_index"],
            "transition_class": selected["transition_class"],
            "policy_role": trace["policy_role"],
            **row,
          }
        )
    all_traces.extend(traces)
    selections[str(seed)] = {
      "selected": True,
      "selection_rule": "lowest pair_index among formal failure_to_success pairs",
      "selection_changed_after_formal_audit": False,
      "pair_index": int(selected["pair_index"]),
      "evaluation_seed": int(selected["evaluation_seed"]),
      "environment_id": int(selected["environment_id"]),
      "formal_transition_class": selected["transition_class"],
      "traces": traces,
    }

  output_csv = output_dir / "mechanism_metrics.csv"
  _write_csv(output_csv, MECHANISM_FIELDS, combined_rows)
  outcome_matches = sum(
    bool(trace["outcome_matches_formal"]) for trace in all_traces
  )
  result = {
    "schema_version": 2,
    "specialist_mode": args.mode,
    "evidence_role": (
      "descriptive first-attempt same-initial-state telemetry replays; "
      "formal paired CSV remains authoritative for episode outcomes"
    ),
    "selection_scope": "formal target paired audit only",
    "selection_rule": (
      "for each seed select the lowest pair_index failure_to_success pair; "
      "never replace an identity or retry until its outcome matches"
    ),
    "paired_audit_csv": str(paired_csv),
    "paired_audit_csv_sha256": _sha256(paired_csv),
    "telemetry_amendment": amendment,
    "selections": selections,
    "trace_reproduction": {
      "trace_count": len(all_traces),
      "outcome_match_count": outcome_matches,
      "outcome_mismatch_count": len(all_traces) - outcome_matches,
      "all_initial_state_signatures_match_formal": all(
        trace["initial_state_signature_matches_formal"]
        for trace in all_traces
      ),
      "all_actor_hashes_match_formal": all(
        trace["actor_state_sha256_matches_formal"] for trace in all_traces
      ),
      "first_attempt_reused_count": sum(
        bool(trace["first_attempt_reused"]) for trace in all_traces
      ),
      "selection_changed_after_formal_audit": False,
      "retry_until_outcome_matches": False,
      "formal_outcomes_replaced_by_replay": False,
    },
    "mechanism_metrics": {
      "path": str(output_csv),
      "sha256": _sha256(output_csv),
      "row_count": len(combined_rows),
    },
  }
  selection_path = output_dir / "mechanism_selection.json"
  temporary = selection_path.with_name(f".{selection_path.name}.tmp")
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(selection_path)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
