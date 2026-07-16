"""Load an rsl_rl checkpoint and execute one deterministic MJLab step."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-CBF")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--expected-actions", type=int, default=12)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  base_env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location="cuda:0",
  )
  policy = runner.get_inference_policy(device="cuda:0")
  obs = env.get_observations()
  with torch.inference_mode():
    action = policy(obs)
    next_obs, reward, done, _ = env.step(action)
  result = {
    "checkpoint": str(args.checkpoint.resolve()),
    "action_shape": list(action.shape),
    "action_finite": bool(torch.isfinite(action).all()),
    "observation_finite": all(torch.isfinite(v).all().item() for v in next_obs.values()),
    "reward": reward.tolist(),
    "done": done.tolist(),
  }
  env.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  if result["action_shape"] != [1, args.expected_actions] or not result["action_finite"]:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
