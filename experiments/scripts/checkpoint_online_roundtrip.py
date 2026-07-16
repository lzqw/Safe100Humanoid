"""Strict reload check for a 799-D online-refinement checkpoint."""

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
  parser.add_argument("--task", default="Unitree-G1-Stairs-Online-DQ")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = "cuda:0"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("online task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    str(args.checkpoint.resolve()),
    load_cfg={"actor": True, "critic": True, "optimizer": True},
    strict=True,
    map_location=device,
  )
  obs, _ = env.reset()
  with torch.inference_mode():
    action = runner.alg.actor(obs)
    value = runner.alg.critic(obs)
  result = {
    "checkpoint": str(args.checkpoint.resolve()),
    "task": args.task,
    "actor_obs_dim": runner.alg.actor.obs_dim,
    "critic_obs_dim": runner.alg.critic.obs_dim,
    "action_shape": list(action.shape),
    "value_shape": list(value.shape),
    "finite": bool(torch.isfinite(action).all() and torch.isfinite(value).all()),
  }
  env.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["finite"] or result["actor_obs_dim"] != 405 or result["critic_obs_dim"] != 799:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
