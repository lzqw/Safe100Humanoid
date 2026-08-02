"""Run one real GPU rollout/update and audit the online PPO action paths."""

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
  parser.add_argument("--safe-bc-weight", type=float, default=0.0)
  parser.add_argument("--runtime-filter", choices=("on", "off"), default="on")
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
  env_cfg.actions["joint_pos"].enabled = args.runtime_filter == "on"
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.num_steps_per_env = args.steps
  agent_cfg.algorithm.num_mini_batches = 1
  agent_cfg.algorithm.num_learning_epochs = 1
  agent_cfg.algorithm.safe_bc_weight = args.safe_bc_weight
  agent_cfg.algorithm.use_counterfactual_cbf_credit = args.runtime_filter == "off"
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("online task must register OnlineSafeRefinementRunner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
  payload = torch.load(
    args.base_checkpoint.resolve(), map_location="cpu", weights_only=False
  )
  source_critic_width = int(payload["critic_state_dict"]["mlp.0.weight"].shape[1])
  if source_critic_width == 283:
    warm_start = runner.load_base_checkpoint(
      str(args.base_checkpoint.resolve()), map_location=device
    )
  else:
    warm_start = runner.load_online_checkpoint(
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
    advantage = runner.alg.shape_intervention_advantages()
  losses = runner.alg.update()
  result = {
    "task": args.task,
    "runtime_filter": args.runtime_filter,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "actor_obs_dim": runner.alg.actor.obs_dim,
    "critic_obs_dim": runner.alg.critic.obs_dim,
    "warm_start": warm_start,
    "credit": credit,
    "advantage": advantage,
    "loss": losses,
    "finite": runner.parameters_are_finite(),
  }
  env.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  if (
    not result["finite"]
    or result["actor_obs_dim"] != 405
    or result["critic_obs_dim"] != 838
    or result["credit"]["policy_storage_max_abs_error"] > 1.0e-6
    or result["credit"]["executed_action_routing_max_abs_error"] > 1.0e-5
    or result["loss"]["policy_old_log_prob_max_abs_error"] > 2.0e-4
    or result["loss"]["policy_old_distribution_param_max_abs_error"] > 1.0e-5
    or not torch.isfinite(torch.tensor(result["loss"]["base_anchor_kl_after_update"]))
  ):
    raise SystemExit(2)


if __name__ == "__main__":
  main()
