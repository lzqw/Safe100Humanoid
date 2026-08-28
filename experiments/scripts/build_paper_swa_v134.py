"""Build the single predeclared v134 actor-SWA deployment checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from proximal_v23_io import actor_state_sha256, file_sha256
from src.tasks.stairs_cbf.paper_uniform_swa_v134 import (
  METHOD_ID,
  V132_SNAPSHOT_SHA256,
  uniform_actor_mlp_average,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument(
    "--checkpoint", type=Path, action="append", required=True
  )
  parser.add_argument("--output-checkpoint", type=Path, required=True)
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
  checkpoints = tuple(path.resolve() for path in args.checkpoint)
  output_checkpoint = args.output_checkpoint.resolve()
  output_manifest = args.output_manifest.resolve()
  if len(checkpoints) != len(V132_SNAPSHOT_SHA256):
    raise ValueError("v134 requires exactly the seven v132 round-01..07 snapshots")
  if len(set(checkpoints)) != len(checkpoints):
    raise ValueError("v134 checkpoint paths must be unique")
  if output_checkpoint.exists() or output_manifest.exists():
    raise FileExistsError("v134 output already exists")
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v134 builder requires a clean committed worktree")

  observed_sha256 = tuple(file_sha256(path) for path in checkpoints)
  if observed_sha256 != V132_SNAPSHOT_SHA256:
    raise RuntimeError(
      "v134 source checkpoint identities differ: "
      f"{observed_sha256} != {V132_SNAPSHOT_SHA256}"
    )
  payloads = [
    torch.load(path, map_location="cpu", weights_only=False)
    for path in checkpoints
  ]
  actor_states = [payload["actor_state_dict"] for payload in payloads]
  averaged_actor, diagnostics = uniform_actor_mlp_average(actor_states)

  output = copy.deepcopy(payloads[-1])
  output["actor_state_dict"] = averaged_actor
  for key in (
    "optimizer_state_dict",
    "proximal_actor_optimizer_state_dict",
    "proximal_critic_optimizer_state_dict",
    "proximal_round_reference_state_dict",
  ):
    output.pop(key, None)
  output.update(
    {
      "proximal_method_id": METHOD_ID,
      "v134_uniform_actor_swa": True,
      "v134_source_checkpoint_sha256": list(observed_sha256),
      "v134_source_rounds": list(range(1, 8)),
      "v134_deployment_only_checkpoint": True,
      "v134_resume_training_supported": False,
      **diagnostics,
    }
  )
  output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_checkpoint.with_name(f".{output_checkpoint.name}.tmp")
  torch.save(output, temporary)
  temporary.replace(output_checkpoint)

  manifest = {
    "schema_version": 1,
    "method_id": METHOD_ID,
    "git_commit": _git(repo, "rev-parse", "HEAD"),
    "source_checkpoints": [str(path) for path in checkpoints],
    "source_checkpoint_sha256": list(observed_sha256),
    "source_rounds": list(range(1, 8)),
    "uniform_snapshot_weight": 1.0 / len(checkpoints),
    "averaged_actor_sha256": actor_state_sha256(averaged_actor),
    "output_checkpoint": str(output_checkpoint),
    "output_checkpoint_sha256": file_sha256(output_checkpoint),
    "deployment_only_checkpoint": True,
    "resume_training_supported": False,
    "diagnostics": diagnostics,
  }
  _atomic_json(output_manifest, manifest)
  print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
