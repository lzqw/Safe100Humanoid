"""Create one provenance-locked v35 actor interpolation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from proximal_v23_io import actor_state_sha256, file_sha256
from src.tasks.stairs_cbf.online import backtrack_actor_state


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--base-sha256", required=True)
  parser.add_argument("--candidate-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-sha256", required=True)
  parser.add_argument("--fraction", type=float, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _required_sha256(path: Path, expected: str) -> str:
  expected = expected.strip().lower()
  if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
    raise ValueError("expected checkpoint SHA-256 must contain 64 hex digits")
  observed = file_sha256(path)
  if observed != expected:
    raise RuntimeError(f"checkpoint SHA-256 mismatch: {observed} != {expected}")
  return observed


def main() -> None:
  args = _parse_args()
  base_path = args.base_checkpoint.resolve()
  candidate_path = args.candidate_checkpoint.resolve()
  output_dir = args.output_dir.resolve()
  if not base_path.is_file() or not candidate_path.is_file():
    raise FileNotFoundError("both v35 interpolation checkpoints must exist")
  if output_dir.exists():
    raise FileExistsError(output_dir)
  if not 0.0 <= args.fraction <= 1.0:
    raise ValueError("v35 model-soup fraction must lie in [0, 1]")
  base_sha = _required_sha256(base_path, args.base_sha256)
  candidate_sha = _required_sha256(candidate_path, args.candidate_sha256)

  base = torch.load(base_path, map_location="cpu", weights_only=False)
  candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
  for name, payload in (("base", base), ("candidate", candidate)):
    if not isinstance(payload.get("actor_state_dict"), dict):
      raise ValueError(f"{name} checkpoint has no actor_state_dict")
  base_actor = base["actor_state_dict"]
  candidate_actor = candidate["actor_state_dict"]
  interpolated = backtrack_actor_state(
    base_actor, candidate_actor, float(args.fraction)
  )
  if not all(bool(torch.isfinite(value).all()) for value in interpolated.values()):
    raise RuntimeError("v35 interpolated actor contains non-finite values")

  output_dir.mkdir(parents=True)
  output_checkpoint = output_dir / "actor.pt"
  temporary_checkpoint = output_dir / ".actor.pt.tmp"
  payload = dict(candidate)
  payload["actor_state_dict"] = {
    key: value.detach().cpu() for key, value in interpolated.items()
  }
  payload["v35_actor_interpolation"] = {
    "method": "trainable_mlp_linear_interpolation",
    "fraction_from_base_to_candidate": float(args.fraction),
    "base_checkpoint_sha256": base_sha,
    "candidate_checkpoint_sha256": candidate_sha,
    "base_actor_sha256": actor_state_sha256(base_actor),
    "candidate_actor_sha256": actor_state_sha256(candidate_actor),
    "non_mlp_state_source": "candidate",
  }
  torch.save(payload, temporary_checkpoint)
  temporary_checkpoint.replace(output_checkpoint)
  summary = {
    "schema_version": 1,
    "experiment": "paper_dual_v35_actor_soup",
    **payload["v35_actor_interpolation"],
    "output_checkpoint": str(output_checkpoint),
    "output_checkpoint_sha256": file_sha256(output_checkpoint),
    "output_actor_sha256": actor_state_sha256(interpolated),
  }
  _atomic_json(output_dir / "interpolation_summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
