"""Build deterministic v20 curve tables, historical references, and manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  FORMAL_ROUNDS,
  SPECIALIST_MODES,
)
from specialist_v20_tables import (
  CANDIDATE_FIELDS,
  REPLAY_FIELDS,
  ROUND_FIELDS,
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    return list(reader.fieldnames or []), list(reader)


def _write_csv(
  path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _combine_training_tables(
  training_root: Path, output_root: Path
) -> dict[str, int]:
  definitions = {
    "round_metrics.csv": (ROUND_FIELDS, 9),
    "candidate_metrics.csv": (CANDIDATE_FIELDS, 24),
    "replay_metrics.csv": (REPLAY_FIELDS, 16),
  }
  counts: dict[str, int] = {}
  for name, (fields, rows_per_run) in definitions.items():
    combined: list[dict[str, str]] = []
    for mode in SPECIALIST_MODES:
      for seed in FORMAL_ADAPTATION_SEEDS:
        path = training_root / mode / f"seed{seed}" / name
        actual_fields, rows = _read_csv(path)
        if actual_fields != fields or len(rows) != rows_per_run:
          raise RuntimeError(f"v20 training table differs: {path}")
        combined.extend(rows)
    expected = len(SPECIALIST_MODES) * len(FORMAL_ADAPTATION_SEEDS) * rows_per_run
    if len(combined) != expected:
      raise RuntimeError(f"combined v20 {name} row count differs")
    _write_csv(output_root / "curves" / name, fields, combined)
    counts[name] = len(combined)
  return counts


def _combine_mechanism_tables(
  audit_root: Path, output_root: Path
) -> int:
  combined: list[dict[str, str]] = []
  fields: list[str] | None = None
  for mode in SPECIALIST_MODES:
    path = audit_root / mode / "mechanism_metrics.csv"
    actual_fields, rows = _read_csv(path)
    if fields is None:
      fields = actual_fields
    elif actual_fields != fields:
      raise RuntimeError("v20 mechanism metric schemas differ")
    combined.extend(rows)
  _write_csv(
    output_root / "curves" / "mechanism_metrics.csv",
    fields or [],
    combined,
  )
  return len(combined)


def _historical_rows(repo: Path) -> list[dict[str, Any]]:
  v17_path = repo / (
    "results/online/specialist_v17/formal/audit/final_audit_compact.json"
  )
  v18_path = repo / (
    "results/online/specialist_v18/formal/diagonal_audit_summary.json"
  )
  v19_path = repo / (
    "results/online/specialist_v19/training/revision4_training_stop.json"
  )
  v17 = json.loads(v17_path.read_text())
  v18 = json.loads(v18_path.read_text())
  rows: list[dict[str, Any]] = []
  for source_mode, label in (("lateral", "lateral"), ("balance", "balance")):
    rows.append(
      {
        "version": "v17",
        "specialist": label,
        "status": "fresh formal paired audit",
        "target_success_delta_pp": 100.0
        * v17["scene_gates"][source_mode]["diagonal_paired_success_delta"],
        "source": str(v17_path.relative_to(repo)),
        "directly_comparable_to_v20": False,
        "note": "context, algorithm, seed count, and audit randomness differ",
      }
    )
  for source_mode, label in (("lateral", "lateral"), ("balance", "balance")):
    rows.append(
      {
        "version": "v18",
        "specialist": label,
        "status": "fresh formal paired audit",
        "target_success_delta_pp": 100.0
        * v18["independent_claims"][source_mode]["target"][
          "paired_success_delta_mean_lcb95_ucb95"
        ][0],
        "source": str(v18_path.relative_to(repo)),
        "directly_comparable_to_v20": False,
        "note": "context, algorithm, seed count, and audit randomness differ",
      }
    )
  rows.extend(
    [
      {
        "version": "v19 R4",
        "specialist": "lateral",
        "status": "training completed; formal audit not started",
        "target_success_delta_pp": "",
        "source": str(v19_path.relative_to(repo)),
        "directly_comparable_to_v20": False,
        "note": "no formal v19-R4 performance result",
      },
      {
        "version": "v19 R4",
        "specialist": "contact_stability",
        "status": "formal training stopped; formal audit not started",
        "target_success_delta_pp": "",
        "source": str(v19_path.relative_to(repo)),
        "directly_comparable_to_v20": False,
        "note": "no formal v19-R4 performance result",
      },
    ]
  )
  return rows


def _artifact_manifest(
  training_root: Path, audit_root: Path
) -> list[dict[str, Any]]:
  records: list[dict[str, Any]] = []
  for scope, root in (("training", training_root), ("audit", audit_root)):
    for path in sorted(root.rglob("*")):
      if not path.is_file():
        continue
      records.append(
        {
          "scope": scope,
          "path": str(path),
          "bytes": path.stat().st_size,
          "sha256": _sha256(path),
        }
      )
  return records


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--audit-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  training_root = args.training_root.resolve()
  audit_root = args.audit_root.resolve()
  output_root = args.output_root.resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  table_counts = _combine_training_tables(training_root, output_root)
  table_counts["mechanism_metrics.csv"] = _combine_mechanism_tables(
    audit_root, output_root
  )
  historical_fields = [
    "version",
    "specialist",
    "status",
    "target_success_delta_pp",
    "source",
    "directly_comparable_to_v20",
    "note",
  ]
  _write_csv(
    output_root / "historical_reference.csv",
    historical_fields,
    _historical_rows(repo),
  )
  manifest = {
    "schema_version": 1,
    "formal_rounds_per_run": FORMAL_ROUNDS,
    "specialist_modes": list(SPECIALIST_MODES),
    "adaptation_seeds": list(FORMAL_ADAPTATION_SEEDS),
    "combined_table_row_counts": table_counts,
    "external_artifacts": _artifact_manifest(training_root, audit_root),
  }
  path = output_root / "training_manifest.json"
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
