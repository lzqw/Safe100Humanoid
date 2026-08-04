"""GPU smoke test for fixed-bank KL diagnostics and transactional restore."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--d0-bank", type=Path, required=True)
  parser.add_argument("--neighbor-bank", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-Online-DQH")
  parser.add_argument("--neighbor-domain", default="DQNH")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-json", type=Path)
  args = parser.parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.algorithm.task_first_constrained = True
  agent_cfg.algorithm.base_anchor_weight = 0.0
  agent_cfg.algorithm.d0_retention_anchor_weight = 0.02
  agent_cfg.algorithm.neighbor_retention_anchor_weight = 0.01
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("retention smoke test requires the online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load_online_checkpoint(
    str(args.checkpoint.resolve()), map_location=args.device
  )
  d0_payload = torch.load(
    args.d0_bank.resolve(), map_location="cpu", weights_only=False
  )
  neighbor_payload = torch.load(
    args.neighbor_bank.resolve(), map_location="cpu", weights_only=False
  )
  metadata = runner.alg.set_retention_anchor_banks(
    d0_payload=d0_payload,
    neighbor_payload=neighbor_payload,
    neighbor_domain=args.neighbor_domain,
  )
  snapshot = runner.snapshot_candidate_state()
  expected_actor = {
    key: value.detach().clone() for key, value in snapshot["actor"].items()
  }
  expected_weights = (
    runner.alg.d0_retention_anchor_weight,
    runner.alg.neighbor_retention_anchor_weight,
  )
  expected_cursors = dict(runner.alg.retention_anchor_cursors)
  kl_metrics = runner.alg.retention_anchor_kl_metrics()
  with torch.no_grad():
    next(runner.alg.actor.parameters()).add_(0.01)
  runner.alg.d0_retention_anchor_weight = 0.19
  runner.alg.neighbor_retention_anchor_weight = 0.18
  runner.alg.retention_anchor_cursors["d0"] = 19
  runner.alg.retention_anchor_cursors["neighbor"] = 23
  runner.restore_candidate_state(snapshot)
  actor_restored = all(
    torch.equal(value, expected_actor[key])
    for key, value in runner.alg.actor.state_dict().items()
  )
  weights_restored = expected_weights == (
    runner.alg.d0_retention_anchor_weight,
    runner.alg.neighbor_retention_anchor_weight,
  )
  cursors_restored = expected_cursors == runner.alg.retention_anchor_cursors
  reference_frozen = runner.alg.retention_actor_reference is not None and all(
    not parameter.requires_grad
    for parameter in runner.alg.retention_actor_reference.parameters()
  )
  result = {
    "passed": bool(
      actor_restored
      and weights_restored
      and cursors_restored
      and reference_frozen
    ),
    "actor_restored": actor_restored,
    "weights_restored": weights_restored,
    "cursors_restored": cursors_restored,
    "reference_frozen": reference_frozen,
    "kl_metrics": kl_metrics,
    "bank_observation_sha256": {
      name: bank["observation_sha256"] for name, bank in metadata.items()
    },
  }
  env.close()
  if not result["passed"]:
    raise RuntimeError(json.dumps(result, sort_keys=True))
  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
