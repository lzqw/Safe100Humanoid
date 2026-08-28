"""Select the single v136 checkpoint from frozen v132 training records."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from src.tasks.stairs_cbf.paper_early_peak_v136 import (
  METHOD_ID,
  V132_EARLIEST_PEAK_CHECKPOINT_SHA256,
  V132_ROUND_METRICS_SHA256,
  earliest_exact_peak_decision,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--round-metrics", type=Path, required=True)
  parser.add_argument("--output-manifest", type=Path, required=True)
  return parser.parse_args()


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  round_metrics_path = args.round_metrics.resolve()
  output_manifest = args.output_manifest.resolve()
  if output_manifest.exists():
    raise FileExistsError(output_manifest)
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v136 selection requires a clean committed worktree")
  observed_metrics_sha256 = file_sha256(round_metrics_path)
  if observed_metrics_sha256 != V132_ROUND_METRICS_SHA256:
    raise RuntimeError("v136 frozen v132 round metrics identity differs")
  records = json.loads(round_metrics_path.read_text())
  candidates = [
    {
      "rollout_round": int(record["round"]),
      "checkpoint_round": int(record["rollout_checkpoint_round"]),
      "checkpoint": record["rollout_checkpoint"],
      "checkpoint_sha256": record["rollout_checkpoint_sha256"],
      "actor_sha256": record["rollout_actor_sha256"],
      "success_count": int(
        record["metrics"]["rollout_filter_on_success_count"]
      ),
      "episode_count": int(
        record["metrics"]["rollout_filter_on_episode_count"]
      ),
      "mean_reached_riser": float(
        record["metrics"]["rollout_filter_on_mean_reached_riser"]
      ),
    }
    for record in records
  ]
  decision = earliest_exact_peak_decision(candidates)
  selected_checkpoint = Path(decision["checkpoint"])
  observed_checkpoint_sha256 = file_sha256(selected_checkpoint)
  if not (
    observed_checkpoint_sha256 == decision["checkpoint_sha256"]
    == V132_EARLIEST_PEAK_CHECKPOINT_SHA256
  ):
    raise RuntimeError("v136 selected checkpoint identity differs")

  manifest = {
    "schema_version": 1,
    "method_id": METHOD_ID,
    "git_commit": _git(repo, "rev-parse", "HEAD"),
    "source_round_metrics": str(round_metrics_path),
    "source_round_metrics_sha256": observed_metrics_sha256,
    "candidate_count": len(candidates),
    "candidates": candidates,
    "decision": decision,
    "selected_checkpoint_sha256_verified": True,
    "selection_additional_evaluation_count": 0,
  }
  _atomic_json(output_manifest, manifest)
  print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
