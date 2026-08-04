"""Collect a fixed, stage-balanced, actor-only retention observation bank."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def collect_bank(
  policy,
  *,
  task: str,
  bank_size: int,
  num_envs: int,
  seed: int,
  device: str,
  runtime_filter: bool,
  maximum_collection_steps: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
  """Collect only actor observations, balancing them over reached risers."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from src.tasks.stairs_cbf.retention import (
    balanced_stage_quotas,
    interleave_stage_observations,
  )

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.actions["joint_pos"].enabled = runtime_filter
  base_env = ManagerBasedRlEnv(cfg, device=device)
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  obs, _ = env.reset()
  policy.eval()
  action_term = base_env.action_manager.get_term("joint_pos")
  num_stages = int(action_term._edge_x.shape[-1])
  quotas = balanced_stage_quotas(bank_size, num_stages)
  stage_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_stages)]
  stage_counts = [0 for _ in range(num_stages)]
  collection_steps = 0

  try:
    with torch.inference_mode():
      while any(count < quota for count, quota in zip(stage_counts, quotas)):
        if collection_steps >= maximum_collection_steps:
          raise RuntimeError(
            "retention bank collection exhausted --maximum-collection-steps; "
            f"counts={stage_counts}, quotas={list(quotas)}"
          )
        actor_observations = obs["actor"]
        if actor_observations.ndim != 2 or actor_observations.shape[0] != num_envs:
          raise RuntimeError("actor observations are not a [num_envs, actor_dim] tensor")
        root_x = base_env.scene["robot"].data.root_link_pos_w[:, 0:1]
        risers = action_term._edge_x[
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        stages = torch.sum(root_x >= risers, dim=1).clamp_max(num_stages - 1)
        for stage, quota in enumerate(quotas):
          remaining = quota - stage_counts[stage]
          if remaining <= 0:
            continue
          ids = (stages == stage).nonzero(as_tuple=False).flatten()[:remaining]
          if len(ids) == 0:
            continue
          selected = actor_observations.index_select(0, ids)
          selected = selected.detach().to(device="cpu", dtype=torch.float32).clone()
          stage_chunks[stage].append(selected)
          stage_counts[stage] += selected.shape[0]
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        collection_steps += 1
  finally:
    env.close()

  by_stage = [
    torch.cat(chunks, dim=0)[:quota].contiguous()
    for chunks, quota in zip(stage_chunks, quotas, strict=True)
  ]
  generator = torch.Generator(device="cpu").manual_seed(seed + 7_919)
  observations = interleave_stage_observations(
    by_stage,
    generator=generator,
  )
  metadata = {
    "bank_size": observations.shape[0],
    "actor_observation_dim": observations.shape[1],
    "num_stages": num_stages,
    "stage_counts": list(quotas),
    "collection_steps": collection_steps,
    "collection_environment_transitions": collection_steps * num_envs,
  }
  return observations, metadata


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--domain", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--checkpoint-kind",
    choices=("online", "base"),
    default="online",
    help="Use base for a legacy checkpoint that needs online observation expansion.",
  )
  parser.add_argument("--bank-size", type=int, default=24_000)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seed", type=int, default=17_001)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--runtime-filter", choices=("on", "off"), default="on")
  parser.add_argument("--maximum-collection-steps", type=int, default=250_000)
  parser.add_argument("--output-bank", type=Path, required=True)
  parser.add_argument("--output-manifest", type=Path, required=True)
  args = parser.parse_args()
  repo = args.repo.resolve()
  checkpoint = args.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if not 20_000 <= args.bank_size <= 50_000:
    raise ValueError("--bank-size must be in [20000, 50000]")
  if args.num_envs < 1 or args.maximum_collection_steps < 1 or args.seed < 0:
    raise ValueError("collection counts and seed must be positive/non-negative")
  if args.task != f"Unitree-G1-Stairs-Online-{args.domain}":
    raise ValueError("--task and --domain must identify the same fixed domain")
  sys.path.insert(0, str(repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.retention import (
    RETENTION_BANK_KIND,
    RETENTION_BANK_SCHEMA_VERSION,
    actor_observation_sha256,
    validate_retention_observation_bank,
  )

  # Build a one-environment loader so the fixed bank itself, rather than PPO
  # storage for the collection environment, owns the host/device memory.
  loader_cfg = load_env_cfg(args.task, play=True)
  loader_cfg.scene.num_envs = 1
  loader_cfg.seed = args.seed
  loader_env_raw = ManagerBasedRlEnv(loader_cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  loader_env = RslRlVecEnvWrapper(
    loader_env_raw, clip_actions=agent_cfg.clip_actions
  )
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(loader_env, asdict(agent_cfg), device=args.device)
  if args.checkpoint_kind == "base":
    if not hasattr(runner, "load_base_checkpoint"):
      raise RuntimeError("the task runner cannot expand a legacy base checkpoint")
    runner.load_base_checkpoint(str(checkpoint), map_location=args.device)
  else:
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
  policy = runner.get_inference_policy(args.device)
  loader_env.close()

  observations, collection = collect_bank(
    policy,
    task=args.task,
    bank_size=args.bank_size,
    num_envs=args.num_envs,
    seed=args.seed,
    device=args.device,
    runtime_filter=args.runtime_filter == "on",
    maximum_collection_steps=args.maximum_collection_steps,
  )
  payload = {
    "schema_version": RETENTION_BANK_SCHEMA_VERSION,
    "kind": RETENTION_BANK_KIND,
    "domain": args.domain,
    "task": args.task,
    "seed": args.seed,
    "runtime_filter": args.runtime_filter == "on",
    "policy_mode": "deterministic_mean",
    "actor_observation_key": "actor",
    "actor_observation_dim": collection["actor_observation_dim"],
    "contains_privileged_observations": False,
    "num_stages": collection["num_stages"],
    "stage_counts": collection["stage_counts"],
    "ordering": "stage_round_robin_v1",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _file_sha256(checkpoint),
    "observation_sha256": actor_observation_sha256(observations),
    "collection_steps": collection["collection_steps"],
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "observations": observations,
  }
  _, validated_metadata = validate_retention_observation_bank(
    payload,
    expected_actor_dim=collection["actor_observation_dim"],
    expected_domain=args.domain,
  )
  output_bank = args.output_bank.resolve()
  output_manifest = args.output_manifest.resolve()
  output_bank.parent.mkdir(parents=True, exist_ok=True)
  output_manifest.parent.mkdir(parents=True, exist_ok=True)
  temporary_bank = output_bank.with_suffix(output_bank.suffix + ".tmp")
  torch.save(payload, temporary_bank)
  temporary_bank.replace(output_bank)
  manifest = {
    **validated_metadata,
    **collection,
    "bank_file": str(output_bank),
    "bank_file_sha256": _file_sha256(output_bank),
  }
  output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
