"""Collect deterministic lowest-ID repair traces from a completed v20 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from specialist_v20_protocol import FORMAL_ADAPTATION_SEEDS, SPECIALIST_MODES

TELEMETRY_FIELDS = [
  "evaluation_seed",
  "environment_id",
  "step",
  "time_s",
  "centerline_error",
  "heading_error",
  "centerline_error_rate",
  "heading_error_rate",
  "command_vy",
  "command_wz",
  "root_edge_margin",
  "foot_edge_margin",
  "left_contact",
  "right_contact",
  "support_foot",
  "left_slip_speed",
  "right_slip_speed",
  "contact_phase_mismatch",
  "roll_rad",
  "pitch_rad",
  "angular_velocity_norm",
  "cbf_correction_norm",
]
MECHANISM_FIELDS = [
  "specialist",
  "adaptation_seed",
  "pair_index",
  "transition_class",
  "policy_role",
  *TELEMETRY_FIELDS,
]


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    return list(reader.fieldnames or []), list(reader)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--mode", choices=SPECIALIST_MODES, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--audit-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _run_trace(
  *,
  repo: Path,
  mode: str,
  context: Path,
  audit_dir: Path,
  output_dir: Path,
  seed: int,
  selected: dict[str, str],
  policy_role: str,
  device: str,
) -> dict[str, Any]:
  checkpoint = (
    audit_dir
    / "raw"
    / ("baseline" if policy_role == "baseline" else "final")
    / f"seed{seed}"
    / "target"
    / "actor.pt"
  )
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  trace_dir = output_dir / "raw" / f"seed{seed}" / policy_role
  trace_dir.mkdir(parents=True, exist_ok=True)
  output_json = trace_dir / "evaluation.json"
  output_csv = trace_dir / "episodes.csv"
  telemetry_csv = trace_dir / "telemetry.csv"
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
  summary = json.loads(output_json.read_text())
  _, episode_rows = _read_csv(output_csv)
  matching = [
    row
    for row in episode_rows
    if row["evaluation_seed"] == selected["evaluation_seed"]
    and row["environment_id"] == selected["environment_id"]
  ]
  if len(matching) != 1:
    raise RuntimeError("mechanism rerun did not reproduce one selected env")
  expected_success = selected[f"{policy_role}_success"]
  if int(matching[0]["success"] == "True") != int(expected_success):
    raise RuntimeError("mechanism rerun outcome differs from the formal audit")
  return {
    "policy_role": policy_role,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint),
    "actor_state_sha256": summary["actor_state_sha256"],
    "initial_state_signature": summary["initial_state_signature"],
    "evaluation_json": str(output_json),
    "evaluation_json_sha256": _sha256(output_json),
    "episode_csv": str(output_csv),
    "episode_csv_sha256": _sha256(output_csv),
    "telemetry_csv": str(telemetry_csv),
    "telemetry_csv_sha256": _sha256(telemetry_csv),
    "telemetry_rows": summary["mechanism_telemetry_row_count"],
  }


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  mode = args.mode
  context = args.context.resolve()
  audit_dir = args.audit_dir.resolve()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  paired_csv = audit_dir / "paired_episode_metrics.csv"
  _, paired_rows = _read_csv(paired_csv)
  target_repairs = [
    row
    for row in paired_rows
    if row["specialist_mode"] == mode
    and row["evaluation_role"] == "target_diagonal_primary"
    and row["transition_class"] == "failure_to_success"
  ]
  selected_by_seed: dict[int, dict[str, str] | None] = {}
  for seed in FORMAL_ADAPTATION_SEEDS:
    candidates = [
      row for row in target_repairs if int(row["adaptation_seed"]) == seed
    ]
    selected_by_seed[seed] = (
      min(candidates, key=lambda row: int(row["pair_index"]))
      if candidates
      else None
    )

  combined_rows: list[dict[str, Any]] = []
  selections: dict[str, Any] = {}
  for seed, selected in selected_by_seed.items():
    if selected is None:
      selections[str(seed)] = {
        "selected": False,
        "reason": "formal target audit contained no failure_to_success pair",
      }
      continue
    traces = [
      _run_trace(
        repo=repo,
        mode=mode,
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
      raise RuntimeError("mechanism baseline/final initial signatures differ")
    for trace in traces:
      fields, rows = _read_csv(Path(trace["telemetry_csv"]))
      if fields != TELEMETRY_FIELDS:
        raise RuntimeError("mechanism telemetry schemas differ")
      for row in rows:
        combined_rows.append(
          {
            "specialist": mode,
            "adaptation_seed": seed,
            "pair_index": selected["pair_index"],
            "transition_class": selected["transition_class"],
            "policy_role": trace["policy_role"],
            **row,
          }
        )
    selections[str(seed)] = {
      "selected": True,
      "selection_rule": "lowest pair_index among failure_to_success pairs",
      "pair_index": int(selected["pair_index"]),
      "evaluation_seed": int(selected["evaluation_seed"]),
      "environment_id": int(selected["environment_id"]),
      "traces": traces,
    }
  output_csv = output_dir / "mechanism_metrics.csv"
  temporary = output_dir / ".mechanism_metrics.csv.tmp"
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=MECHANISM_FIELDS)
    writer.writeheader()
    writer.writerows(combined_rows)
  temporary.replace(output_csv)
  result = {
    "schema_version": 1,
    "specialist_mode": mode,
    "selection_scope": "formal target paired audit only",
    "selection_rule": (
      "for each seed select the lowest pair_index failure_to_success pair; "
      "never select by visual appearance"
    ),
    "paired_audit_csv": str(paired_csv),
    "paired_audit_csv_sha256": _sha256(paired_csv),
    "selections": selections,
    "mechanism_metrics": {
      "path": str(output_csv),
      "sha256": _sha256(output_csv),
      "row_count": len(combined_rows),
    },
  }
  selection_path = output_dir / "mechanism_selection.json"
  selection_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
