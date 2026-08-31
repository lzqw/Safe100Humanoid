"""Re-evaluate one saved task-first candidate with a larger paired GPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch


class _CheckpointAlgorithm:
  """Minimal runner algorithm interface required by ``_evaluate_state``."""

  def __init__(self, payload: dict[str, Any]) -> None:
    self._payload = payload

  def save(self) -> dict[str, Any]:
    # _evaluate_state replaces only top-level actor/metadata entries before
    # serializing its temporary checkpoint.  A shallow copy keeps the source
    # checkpoint immutable without duplicating all critic tensors in memory.
    return dict(self._payload)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _actor_is_finite(payload: dict[str, Any]) -> bool:
  state = payload.get("actor_state_dict")
  return isinstance(state, dict) and all(
    bool(torch.isfinite(value).all()) for value in state.values()
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--old-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-checkpoint", type=Path, required=True)
  parser.add_argument("--online-summary", type=Path, required=True)
  parser.add_argument("--baseline-eval", type=Path, required=True)
  parser.add_argument("--round", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--target-domain", default="DQH")
  parser.add_argument("--retention-domain", default="D0")
  parser.add_argument("--neighbor-domain", default="DQNH")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--repeats", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--runtime-filter",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument("--maximum-target-fall-rate", type=float, default=1.0)
  args = parser.parse_args()

  if args.round < 1:
    raise ValueError("--round is one-based and must be positive")
  if args.num_envs < 1 or args.repeats < 1:
    raise ValueError("paired audit environment/repeat counts must be positive")
  for path in (
    args.old_checkpoint,
    args.candidate_checkpoint,
    args.online_summary,
    args.baseline_eval,
  ):
    if not path.is_file():
      raise FileNotFoundError(path)

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  from experiments.scripts.online_refine_stairs import _evaluate_state
  from src.tasks.stairs_cbf.online import (
    CandidateGateThresholds,
    candidate_gate,
    candidate_gate_intervals,
    safe_improvement_score,
  )

  old_payload = torch.load(
    args.old_checkpoint.resolve(), map_location="cpu", weights_only=False
  )
  candidate_payload = torch.load(
    args.candidate_checkpoint.resolve(), map_location="cpu", weights_only=False
  )
  if not _actor_is_finite(old_payload) or not _actor_is_finite(candidate_payload):
    raise RuntimeError("old or candidate actor checkpoint is non-finite")
  if old_payload["actor_state_dict"].keys() != candidate_payload[
    "actor_state_dict"
  ].keys():
    raise RuntimeError("old and candidate actor state keys differ")
  old_checkpoint_sha256 = _sha256(args.old_checkpoint.resolve())
  candidate_checkpoint_sha256 = _sha256(
    args.candidate_checkpoint.resolve()
  )

  online = json.loads(args.online_summary.read_text())
  rounds = online if isinstance(online, list) else online["rounds"]
  if args.round > len(rounds):
    raise ValueError(f"round {args.round} is absent from the online summary")
  round_result = rounds[args.round - 1]
  if int(round_result["round"]) != args.round:
    raise RuntimeError("online round ledger is not ordered by round number")
  baseline = json.loads(args.baseline_eval.read_text())

  domains = (
    args.retention_domain,
    args.target_domain,
    args.neighbor_domain,
  )
  replicate_root = args.output.resolve().parent / "replicates"
  runner = SimpleNamespace(alg=_CheckpointAlgorithm(old_payload))
  old_eval = _evaluate_state(
    runner,
    old_payload["actor_state_dict"],
    domains=domains,
    num_envs=args.num_envs,
    num_episodes=args.num_envs,
    seed=args.seed,
    device=args.device,
    repeats=args.repeats,
    runtime_filter=args.runtime_filter,
    artifact_dir=replicate_root / f"old-{old_checkpoint_sha256[:16]}",
    resume=True,
  )
  candidate_eval = _evaluate_state(
    runner,
    candidate_payload["actor_state_dict"],
    domains=domains,
    num_envs=args.num_envs,
    num_episodes=args.num_envs,
    seed=args.seed,
    device=args.device,
    repeats=args.repeats,
    runtime_filter=args.runtime_filter,
    artifact_dir=(
      replicate_root / f"candidate-{candidate_checkpoint_sha256[:16]}"
    ),
    resume=True,
  )

  thresholds = CandidateGateThresholds(
    maximum_target_fall_rate=args.maximum_target_fall_rate,
    require_task_improvement=True,
  )
  old_total_kl = float(round_result["old_total_kl_from_base"])
  candidate_total_kl = float(round_result["total_kl_from_base"])
  accepted, reasons = candidate_gate(
    update_metrics=round_result["update_metrics"],
    old_eval=old_eval,
    candidate_eval=candidate_eval,
    base_d0_success=float(baseline[args.retention_domain]["success_rate"]),
    old_total_kl_from_base=old_total_kl,
    total_kl_from_base=candidate_total_kl,
    parameters_finite=True,
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
    retention_domain=args.retention_domain,
    neighbor_domain=args.neighbor_domain,
    old_total_kl_from_base=old_total_kl,
    total_kl_from_base=candidate_total_kl,
  )
  result = {
    "accepted": accepted,
    "decision": "accept" if accepted else "rollback",
    "rejection_reasons": reasons,
    "round": args.round,
    "selected_candidate_fraction": round_result.get(
      "selected_candidate_fraction"
    ),
    "target_domain": args.target_domain,
    "retention_domain": args.retention_domain,
    "neighbor_domain": args.neighbor_domain,
    "num_envs": args.num_envs,
    "repeats": args.repeats,
    "episodes_per_domain_per_policy": args.num_envs * args.repeats,
    "seed": args.seed,
    "runtime_filter": args.runtime_filter,
    "old_checkpoint": str(args.old_checkpoint.resolve()),
    "old_checkpoint_sha256": old_checkpoint_sha256,
    "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
    "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
    "old_total_kl_from_base": old_total_kl,
    "total_kl_from_base": candidate_total_kl,
    "gate_intervals": intervals,
    "safe_improvement_scores": {
      "old": safe_improvement_score(
        old_eval[args.target_domain], total_kl_from_base=old_total_kl
      ),
      "candidate": safe_improvement_score(
        candidate_eval[args.target_domain],
        total_kl_from_base=candidate_total_kl,
      ),
    },
    "old_eval": old_eval,
    "candidate_eval": candidate_eval,
    "update_metrics": round_result["update_metrics"],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
