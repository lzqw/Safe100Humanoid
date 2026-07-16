"""One real GPU rollout/update check for the expanded online PPO stack."""

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
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-Online-DQ")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--safe-bc-weight", type=float, default=0.05)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.utils import check_nan

  device = "cuda:0"
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.num_steps_per_env = args.steps
  agent_cfg.algorithm.num_mini_batches = 1
  agent_cfg.algorithm.num_learning_epochs = 1
  agent_cfg.algorithm.safe_bc_weight = args.safe_bc_weight
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("online task must register OnlineSafeRefinementRunner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
  warm_start = runner.load_base_checkpoint(
    str(args.base_checkpoint.resolve()), map_location=device
  )
  obs, _ = env.reset()
  runner.alg.train_mode()
  runner.alg.clear_cbf_rollout()
  with torch.no_grad():
    for _ in range(args.steps):
      action = runner.alg.act(obs)
      obs, reward, done, extras = env.step(action)
      check_nan(obs, reward, done)
      runner.alg.process_env_step(obs, reward, done, extras)
    credit = runner.alg.relabel_pre_intervention_costs()
    runner.alg.compute_returns(obs)
  losses = runner.alg.update()
  result = {
    "task": args.task,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "actor_obs_dim": runner.alg.actor.obs_dim,
    "critic_obs_dim": runner.alg.critic.obs_dim,
    "warm_start": warm_start,
    "credit": credit,
    "loss": losses,
    "finite": runner.parameters_are_finite(),
  }
  env.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["finite"] or result["actor_obs_dim"] != 405 or result["critic_obs_dim"] != 799:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
