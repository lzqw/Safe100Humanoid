"""Create a policy-line-search checkpoint from one conservative PPO step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-checkpoint", type=Path, required=True)
  parser.add_argument("--fraction", type=float, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  from src.tasks.stairs_cbf.online import backtrack_actor_state

  base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
  candidate = torch.load(
    args.candidate_checkpoint, map_location="cpu", weights_only=False
  )
  candidate["actor_state_dict"] = backtrack_actor_state(
    base["actor_state_dict"], candidate["actor_state_dict"], args.fraction
  )
  metadata = {
    "kind": "ppo_actor_backtracking",
    "fraction": args.fraction,
    "base_checkpoint": str(args.base_checkpoint),
    "candidate_checkpoint": str(args.candidate_checkpoint),
  }
  infos = candidate.setdefault("infos", {})
  if not isinstance(infos, dict):
    infos = {"source_infos": infos}
    candidate["infos"] = infos
  infos["line_search"] = metadata
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(candidate, args.output)
  print(json.dumps({**metadata, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
  main()
