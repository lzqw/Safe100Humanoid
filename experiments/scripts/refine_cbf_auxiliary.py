"""Distill true CBF interventions with one explicitly bounded auxiliary step."""

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
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-Online-DQ")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=768)
  parser.add_argument("--learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.utils import check_nan

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task)
  agent_cfg.num_steps_per_env = args.steps
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("online task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load(
    str(args.checkpoint.resolve()),
    load_cfg={"actor": True, "critic": True, "optimizer": True},
    strict=True,
    map_location=args.device,
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
  auxiliary = runner.alg.apply_safe_bc_auxiliary(learning_rate=args.learning_rate)
  result = {
    "task": args.task,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "seed": args.seed,
    "source_checkpoint": str(args.checkpoint.resolve()),
    "output_checkpoint": str(args.output.resolve()),
    "intervention_fraction": float(runner.alg.cbf_intervened.float().mean()),
    "correction_mean": float(runner.alg.cbf_magnitude.mean()),
    "auxiliary": auxiliary,
    "parameters_finite": runner.parameters_are_finite(),
  }
  payload = runner.alg.save()
  payload["infos"] = {"cbf_auxiliary": result}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, args.output)
  args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  env.close()
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["parameters_finite"]:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
