"""Freeze the dominant v15 baseline failure class for Branch-B v16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument(
    "--input-root",
    type=Path,
    required=True,
    help="v15 final-audit root containing train_seed*/baseline/DQHMED CSVs.",
  )
  parser.add_argument("--source-audit-summary", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  parser.add_argument("--expected-episodes", type=int, default=1536)
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _as_bool(value: str) -> bool:
  if value not in ("True", "False"):
    raise ValueError(f"invalid boolean value in evaluation CSV: {value!r}")
  return value == "True"


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  from src.tasks.stairs_cbf.hard_cases import (
    HIGH_CBF_CORRECTION_THRESHOLD,
    LATERAL_CENTERLINE_WIDTH_FRACTION,
    LATERAL_HEADING_THRESHOLD_RAD,
    classify_target_failure_mode,
  )

  input_root = args.input_root.resolve()
  audit_path = args.source_audit_summary.resolve()
  audit = json.loads(audit_path.read_text())
  if (
    audit.get("method") != "Failure-Focused Brief PPO v15"
    or audit.get("post_v15_decision", {}).get("branch") != "B"
    or audit.get("training_seeds") != [42, 142, 242]
  ):
    raise ValueError("source audit is not the completed three-seed v15 Branch-B audit")

  paths = sorted(
    input_root.glob("train_seed*/baseline/DQHMED/DQHMED-seed*.csv")
  )
  if not paths:
    raise FileNotFoundError("no v15 baseline DQH-Medium episode CSVs found")
  classified: list[dict[str, Any]] = []
  source_episode_count = 0
  source_success_count = 0
  source_fall_count = 0
  file_records: list[dict[str, Any]] = []
  seen_keys: set[tuple[int, int, int]] = set()
  for path in paths:
    training_match = re.search(r"train_seed(\d+)", str(path))
    evaluation_match = re.search(r"DQHMED-seed(\d+)\.csv$", path.name)
    if training_match is None or evaluation_match is None:
      raise ValueError(f"cannot recover seed identities from {path}")
    training_seed = int(training_match.group(1))
    evaluation_seed = int(evaluation_match.group(1))
    if training_seed not in (42, 142, 242):
      raise ValueError(f"unexpected adaptation seed in classification input: {path}")
    row_count = 0
    with path.open(newline="") as handle:
      reader = csv.DictReader(handle)
      required = {
        "episode",
        "success",
        "fell",
        "side_edge_breach",
        "max_abs_centerline_error",
        "max_abs_heading_error",
        "correction_max",
        "max_riser",
      }
      if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"classification input lacks required fields: {path}")
      for row in reader:
        episode = int(row["episode"])
        key = (training_seed, evaluation_seed, episode)
        if key in seen_keys:
          raise ValueError(f"duplicate baseline episode identity: {key}")
        seen_keys.add(key)
        row_count += 1
        source_episode_count += 1
        success = _as_bool(row["success"])
        fell = _as_bool(row["fell"])
        source_success_count += int(success)
        source_fall_count += int(fell)
        if not fell:
          continue
        failure_type = classify_target_failure_mode(
          side_edge_breach=_as_bool(row["side_edge_breach"]),
          max_abs_centerline_error=float(row["max_abs_centerline_error"]),
          max_abs_heading_error=float(row["max_abs_heading_error"]),
          correction_max=float(row["correction_max"]),
        )
        classified.append(
          {
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "episode": episode,
            "failure_type": failure_type,
            "side_edge_breach": _as_bool(row["side_edge_breach"]),
            "max_abs_centerline_error": float(
              row["max_abs_centerline_error"]
            ),
            "max_abs_heading_error": float(row["max_abs_heading_error"]),
            "correction_max": float(row["correction_max"]),
            "max_riser": int(row["max_riser"]),
          }
        )
    file_records.append(
      {
        "path": str(path.relative_to(input_root)),
        "sha256": _sha256(path),
        "row_count": row_count,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
      }
    )

  if source_fall_count != len(classified):
    raise RuntimeError("every v15 baseline fall must receive exactly one class")
  if not args.smoke and source_episode_count != args.expected_episodes:
    raise ValueError(
      f"formal classification expected {args.expected_episodes} episodes, "
      f"found {source_episode_count}"
    )
  failure_type_counts: dict[str, int] = {}
  for row in classified:
    failure_type = str(row["failure_type"])
    failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
  ordered = sorted(
    failure_type_counts.items(), key=lambda item: (-item[1], item[0])
  )
  if len(ordered) < 2 or ordered[0][1] <= ordered[1][1]:
    raise RuntimeError("v15 evidence does not identify a unique dominant failure type")
  selected = ordered[0][0]

  output_csv = args.output_csv.resolve()
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(classified[0]))
    writer.writeheader()
    writer.writerows(classified)

  result = {
    "method": "Branch-B target failure classification v16",
    "source_method": "Failure-Focused Brief PPO v15",
    "source_policy_role": "online-start baseline",
    "source_audit_summary": {
      "path": str(audit_path),
      "sha256": _sha256(audit_path),
    },
    "source_training_seeds": [42, 142, 242],
    "source_episode_count": source_episode_count,
    "source_success_count": source_success_count,
    "source_fall_count": source_fall_count,
    "source_files": file_records,
    "thresholds": {
      "centerline_width_fraction": LATERAL_CENTERLINE_WIDTH_FRACTION,
      "heading_error_rad": LATERAL_HEADING_THRESHOLD_RAD,
      "heading_error_degrees": math.degrees(LATERAL_HEADING_THRESHOLD_RAD),
      "high_cbf_correction": HIGH_CBF_CORRECTION_THRESHOLD,
      "stair_half_width_m": 1.2,
    },
    "classification_rule": (
      "lateral_heading_drift if side edge breached, root centerline error >= "
      "2/3 stair half-width, or heading error >= 90 degrees; otherwise split "
      "at correction_max >= 0.5"
    ),
    "failure_type_counts": failure_type_counts,
    "failure_type_fractions": {
      key: value / source_fall_count for key, value in failure_type_counts.items()
    },
    "selection_rule": "select the unique largest baseline fall class",
    "selected_dominant_failure_type": selected,
    "adapted_v16_policy_evaluations_used": False,
    "classified_falls_csv": {
      "path": str(output_csv),
      "sha256": _sha256(output_csv),
      "row_count": len(classified),
    },
  }
  output_json = args.output_json.resolve()
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
