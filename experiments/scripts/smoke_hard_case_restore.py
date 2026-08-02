"""GPU integration smoke for pre-CBF hard-case capture and restoration."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


def _row(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
  ids = torch.tensor([index], device=next(iter(state.values())).device)
  return {key: value.index_select(0, ids) for key, value in state.items()}


def _comparison_errors(
  expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> dict[str, float]:
  # Terrain metadata describes the source tile and is intentionally not written
  # into the destination terrain generator.  Every other captured tensor is
  # part of the restored online MDP state.
  ignored = {"terrain/type", "terrain/level"}
  errors: dict[str, float] = {}
  for key in sorted(expected.keys() - ignored):
    lhs = expected[key].detach().cpu()
    rhs = actual[key].detach().cpu()
    if lhs.dtype == torch.bool:
      errors[key] = float(torch.logical_xor(lhs, rhs).sum())
    else:
      errors[key] = float((lhs.to(torch.float64) - rhs.to(torch.float64)).abs().max())
  return errors


def _assert_observation_finite(obs) -> None:
  for key, value in obs.items():
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
      raise RuntimeError(f"non-finite observation group: {key}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-Online-DQH")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--warmup-steps", type=int, default=96)
  parser.add_argument("--event-steps", type=int, default=1024)
  parser.add_argument("--pre-steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.utils import check_nan
  from src.tasks.stairs_cbf.hard_cases import (
    HardCaseStateBank,
    capture_hard_case_state,
    reset_rollout_with_hard_cases,
    restore_hard_case_state,
  )
  from src.tasks.stairs_cbf.online import adaptive_cbf_std_factor

  device = "cuda:0"
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(args.task)
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("online task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load_base_checkpoint(str(args.base_checkpoint.resolve()), map_location=device)
  policy = runner.get_inference_policy(device=device)

  result: dict[str, object] = {
    "task": args.task,
    "num_envs": args.num_envs,
    "pre_steps": args.pre_steps,
  }
  try:
    obs, _ = env.reset()
    with torch.inference_mode():
      for _ in range(args.warmup_steps):
        obs, reward, done, _ = env.step(policy(obs))
        check_nan(obs, reward, done)

      captured = _row(capture_hard_case_state(base_env), 0)
      destination = torch.tensor([1], device=device, dtype=torch.long)
      base_env._reset_idx(destination)
      base_env.scene.write_data_to_sim()
      base_env.sim.forward()
      base_env.sim.sense()
      base_env.observation_manager.compute(update_history=True)
      restore_hard_case_state(base_env, captured, destination)
      base_env.scene.write_data_to_sim()
      base_env.sim.forward()
      for sensor in base_env.scene.sensors.values():
        sensor._invalidate_cache()
      base_env.sim.sense()
      restored = _row(capture_hard_case_state(base_env), 1)
      errors = _comparison_errors(captured, restored)
      max_error = max(errors.values(), default=0.0)
      result["restore_max_abs_error"] = max_error
      result["restore_nonzero_keys"] = {
        key: value for key, value in errors.items() if value > 1.0e-6
      }

      # Build the bank only from genuine intervention rising edges, using the
      # state exactly pre_steps earlier and rejecting histories crossing reset.
      bank = HardCaseStateBank(capacity=64)
      history: deque[dict[str, torch.Tensor]] = deque(maxlen=args.pre_steps + 1)
      valid_age = torch.zeros(args.num_envs, device=device, dtype=torch.long)
      previous_intervened = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
      action_term = base_env.action_manager.get_term("joint_pos")
      for _ in range(args.event_steps):
        history.append(capture_hard_case_state(base_env))
        obs, reward, done, extras = env.step(policy(obs))
        check_nan(obs, reward, done)
        valid_age += 1
        valid_age[done] = 0
        intervened = action_term.intervened.bool()
        rising = intervened & ~previous_intervened & (valid_age >= args.pre_steps)
        previous_intervened = intervened.clone()
        previous_intervened[done] = False
        if len(history) == args.pre_steps + 1 and bool(rising.any()):
          ids = torch.where(rising)[0]
          priority = action_term.intervention_norm.index_select(0, ids).clamp_min(1.0e-6)
          riser = extras["online_stair_index"].index_select(0, ids).to(torch.long)
          bank.add_batched(history[0], ids, priority, riser)
        if len(bank) >= min(4, args.num_envs):
          break

      generator = torch.Generator(device="cpu")
      generator.manual_seed(args.seed + 100003)
      obs, start_metrics = reset_rollout_with_hard_cases(
        env, bank, hard_case_fraction=0.5, generator=generator
      )
      _assert_observation_finite(obs)
      obs, reward, done, _ = env.step(policy(obs))
      check_nan(obs, reward, done)
      result["bank_size"] = len(bank)
      result["bank_total_added"] = bank.total_added
      result["hard_start_metrics"] = start_metrics
      result["post_restore_step_finite"] = True

      actual_demand = float(action_term.intervened.float().mean())
      adaptation = adaptive_cbf_std_factor(
        intervention_per_riser=actual_demand,
        target_intervention_per_riser=0.10,
        adaptation_rate=0.10,
        fall_count=0.0,
      )
      result["adaptive_std_probe"] = {
        "observed_intervention_fraction": actual_demand,
        "factor": adaptation,
      }
  finally:
    env.close()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  if (
    float(result["restore_max_abs_error"]) > 1.0e-6
    or int(result["bank_size"]) < 1
    or int(result["hard_start_metrics"]["hard_case_start_count"]) < 1
    or not result["post_restore_step_finite"]
  ):
    raise SystemExit(2)


if __name__ == "__main__":
  main()
