"""Re-evaluate a saved online candidate with a larger paired GPU batch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch


class _CheckpointAlgorithm:
  def __init__(self, payload: dict) -> None:
    self._payload = payload

  def save(self) -> dict:
    return copy.deepcopy(self._payload)


class _CheckpointRunner:
  def __init__(self, payload: dict) -> None:
    self.alg = _CheckpointAlgorithm(payload)


def _parameters_are_finite(state: dict[str, torch.Tensor]) -> bool:
  return all(bool(torch.isfinite(value).all()) for value in state.values())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--old-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-checkpoint", type=Path, required=True)
  parser.add_argument("--online-rounds", type=Path, required=True)
  parser.add_argument("--round", type=int, default=1)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--retention-domain", default="D0")
  parser.add_argument("--target-domain", default="DQ")
  parser.add_argument("--neighbor-domain", default="DQN")
  parser.add_argument("--base-d0-success", type=float)
  parser.add_argument("--maximum-target-fall-rate", type=float, default=1.0)
  args = parser.parse_args()

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  from experiments.scripts.online_refine_stairs import _evaluate_state
  from src.tasks.stairs_cbf.online import (
    CandidateGateThresholds,
    candidate_gate,
    candidate_gate_intervals,
    safe_improvement_score,
  )

  if args.round < 1:
    raise ValueError("--round is one-based and must be positive")
  if args.num_envs < 1 or args.repeats < 1:
    raise ValueError("--num-envs and --repeats must be positive")
  rounds = json.loads(args.online_rounds.resolve().read_text())
  if isinstance(rounds, dict):
    rounds = rounds["rounds"]
  round_result = rounds[args.round - 1]
  if int(round_result["round"]) != args.round:
    raise ValueError("selected online round does not match --round")

  domains = (
    args.retention_domain,
    args.target_domain,
    args.neighbor_domain,
  )

  def evaluate(checkpoint: Path) -> tuple[dict, bool]:
    payload = torch.load(
      checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    actor_state = payload["actor_state_dict"]
    finite = _parameters_are_finite(actor_state)
    matrix = _evaluate_state(
      _CheckpointRunner(payload),
      actor_state,
      domains=domains,
      num_envs=args.num_envs,
      num_episodes=args.num_envs,
      seed=args.seed,
      device=args.device,
      repeats=args.repeats,
      runtime_filter=True,
    )
    return matrix, finite

  old_eval, old_finite = evaluate(args.old_checkpoint)
  candidate_eval, candidate_finite = evaluate(args.candidate_checkpoint)
  thresholds = CandidateGateThresholds(
    maximum_target_fall_rate=args.maximum_target_fall_rate
  )
  base_d0_success = (
    old_eval[args.retention_domain]["success_rate"]
    if args.base_d0_success is None
    else args.base_d0_success
  )
  accepted, reasons = candidate_gate(
    update_metrics=round_result["update_metrics"],
    old_eval=old_eval,
    candidate_eval=candidate_eval,
    base_d0_success=base_d0_success,
    old_total_kl_from_base=float(
      round_result.get("old_total_kl_from_base", 0.0)
    ),
    total_kl_from_base=float(round_result["total_kl_from_base"]),
    parameters_finite=old_finite and candidate_finite,
    thresholds=thresholds,
    target_domain=args.target_domain,
    retention_domain=args.retention_domain,
    neighbor_domain=args.neighbor_domain,
  )
  intervals = candidate_gate_intervals(
    old_eval=old_eval,
    candidate_eval=candidate_eval,
    thresholds=thresholds,
    target_domain=args.target_domain,
    neighbor_domain=args.neighbor_domain,
    old_total_kl_from_base=float(
      round_result.get("old_total_kl_from_base", 0.0)
    ),
    total_kl_from_base=float(round_result["total_kl_from_base"]),
  )
  result = {
    "accepted": accepted,
    "decision": "accept" if accepted else "rollback",
    "rejection_reasons": reasons,
    "round": args.round,
    "old_checkpoint": str(args.old_checkpoint.resolve()),
    "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
    "num_episodes_per_domain_policy": args.num_envs * args.repeats,
    "old_eval": old_eval,
    "candidate_eval": candidate_eval,
    "gate_intervals": intervals,
    "safe_improvement_scores": {
      "old": safe_improvement_score(
        old_eval[args.target_domain],
        total_kl_from_base=float(
          round_result.get("old_total_kl_from_base", 0.0)
        ),
      ),
      "candidate": safe_improvement_score(
        candidate_eval[args.target_domain],
        total_kl_from_base=float(round_result["total_kl_from_base"]),
      ),
    },
    "update_metrics": round_result["update_metrics"],
    "total_kl_from_base": round_result["total_kl_from_base"],
  }
  args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
  args.output.resolve().write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
