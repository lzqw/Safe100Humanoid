"""Transactional CBF-protected PPO refinement on a fixed deployment stair."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

from evaluate_online_stairs import evaluate_policy


def _actor_state(actor) -> dict[str, torch.Tensor]:
  return {key: value.detach().clone() for key, value in actor.state_dict().items()}


def _evaluate_state(
  runner,
  actor_state: dict[str, torch.Tensor],
  *,
  domains: tuple[str, ...],
  num_envs: int,
  num_episodes: int,
  seed: int,
  device: str,
  repeats: int = 1,
) -> dict[str, dict[str, Any]]:
  original_device = next(runner.alg.actor.parameters()).device
  current = _actor_state(runner.alg.actor)
  output: dict[str, dict[str, Any]] = {}
  try:
    runner.alg.actor.to(device)
    runner.alg.actor.load_state_dict(actor_state, strict=True)
    for domain in domains:
      replicate_summaries = []
      for repeat in range(repeats):
        summary, _ = evaluate_policy(
          runner.alg.actor,
          task=f"Unitree-G1-Stairs-Online-{domain}",
          num_envs=num_envs,
          num_episodes=num_episodes,
          seed=seed + repeat,
          device=device,
        )
        replicate_summaries.append(summary)
      aggregate: dict[str, Any] = {
        "task": f"Unitree-G1-Stairs-Online-{domain}",
        "num_episodes": num_episodes * repeats,
        "repeats": repeats,
        "seeds": [seed + repeat for repeat in range(repeats)],
        "replicates": replicate_summaries,
      }
      for key in (
        "success_rate",
        "fall_rate",
        "timeout_rate",
        "mean_reached_riser",
        "intervention_per_riser",
        "correction_mean",
      ):
        values = [float(summary[key]) for summary in replicate_summaries]
        aggregate[key] = sum(values) / len(values)
        aggregate[f"{key}_std"] = (
          math.sqrt(sum((value - aggregate[key]) ** 2 for value in values) / (len(values) - 1))
          if len(values) > 1
          else 0.0
        )
      output[domain] = aggregate
  finally:
    runner.alg.actor.load_state_dict(current, strict=True)
    runner.alg.actor.to(original_device)
  return output


def _total_actor_kl(runner, base_state: dict[str, torch.Tensor]) -> float:
  obs = runner.alg.storage.observations.flatten(0, 1)
  candidate_state = _actor_state(runner.alg.actor)
  with torch.no_grad():
    runner.alg.actor.load_state_dict(base_state, strict=True)
    base_mean = runner.alg.actor(obs).detach().clone()
    base_std = runner.alg.actor.distribution.std_param.detach().clone()
    runner.alg.actor.load_state_dict(candidate_state, strict=True)
    candidate_mean = runner.alg.actor(obs).detach()
    # Cross-round drift constrains the deterministic deployment behavior.  A
    # separately bounded/reduced exploration std must not by itself exhaust
    # the mean-policy KL budget after a rejected candidate.
    kl = 0.5 * torch.sum(
      ((candidate_mean - base_mean) / base_std.clamp_min(1.0e-6)) ** 2,
      dim=-1,
    ).mean()
  return float(kl)


def _collect_and_update(runner, obs, *, critic_only: bool):
  from rsl_rl.utils import check_nan

  runner.alg.set_critic_only(critic_only)
  runner.alg.clear_cbf_rollout()
  runner.alg.train_mode()
  # Use no_grad rather than inference_mode because critic normalization is
  # intentionally updated and must remain rollback-compatible.
  with torch.no_grad():
    for _ in range(runner.cfg["num_steps_per_env"]):
      actions = runner.alg.act(obs)
      obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
      check_nan(obs, rewards, dones)
      obs = obs.to(runner.device)
      rewards = rewards.to(runner.device)
      dones = dones.to(runner.device)
      runner.alg.process_env_step(obs, rewards, dones, extras)
    credit_metrics = runner.alg.relabel_pre_intervention_costs()
    runner.alg.compute_returns(obs)
  losses = runner.alg.update()
  losses.update(credit_metrics)
  return obs, losses


def _save_checkpoint(runner, path: Path, *, iteration: int, metadata: dict[str, Any]) -> None:
  payload = runner.alg.save()
  payload["iter"] = iteration
  payload["infos"] = {"online_refinement": metadata}
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rollout-steps", type=int, default=256)
  parser.add_argument("--critic-burn-in-rounds", type=int, default=2)
  parser.add_argument("--online-rounds", type=int, default=2)
  parser.add_argument("--eval-num-envs", type=int, default=8)
  parser.add_argument("--eval-num-episodes", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--actor-learning-rate", type=float, default=1.0e-5)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--pre-intervention-weight", type=float, default=0.20)
  parser.add_argument("--std-scale-from-base", type=float, default=0.35)
  parser.add_argument("--safe-bc-weight", type=float, default=0.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--gate-device",
    default="cuda:0",
    help="Physics device for paired candidate acceptance (GPU by default).",
  )
  parser.add_argument(
    "--gate-repeats",
    type=int,
    default=3,
    help="Independent fixed-seed GPU evaluation replicates per policy/domain.",
  )
  parser.add_argument(
    "--train-domain",
    default="DQ",
    help="Target domain used for online rollouts (DQ quick prototype or D4 formal).",
  )
  parser.add_argument("--neighbor-domain", default="DQN")
  parser.add_argument(
    "--baseline-domains",
    nargs="+",
    default=["D0", "D1", "D2", "D3", "D4", "D5", "DQ", "DQN"],
  )
  args = parser.parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.online import (
    CandidateGateThresholds,
    candidate_gate,
    candidate_precheck,
  )

  task = f"Unitree-G1-Stairs-Online-{args.train_domain}"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(task)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  agent_cfg.algorithm.actor_learning_rate = args.actor_learning_rate
  agent_cfg.algorithm.critic_learning_rate = args.critic_learning_rate
  agent_cfg.algorithm.pre_intervention_weight = args.pre_intervention_weight
  agent_cfg.algorithm.std_scale_from_base = args.std_scale_from_base
  agent_cfg.algorithm.safe_bc_weight = args.safe_bc_weight
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_base_checkpoint(
    str(args.base_checkpoint), map_location=args.device
  )
  obs, _ = env.reset()
  base_actor_state = _actor_state(runner.alg.actor)
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  baseline_eval = _evaluate_state(
    runner,
    base_actor_state,
    domains=tuple(args.baseline_domains),
    num_envs=args.eval_num_envs,
    num_episodes=args.eval_num_episodes,
    seed=args.seed,
    device=args.device,
    repeats=args.gate_repeats,
  )
  (output_dir / "baseline_ood_matrix.json").write_text(
    json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n"
  )

  burn_in: list[dict[str, float]] = []
  for _ in range(args.critic_burn_in_rounds):
    obs, metrics = _collect_and_update(runner, obs, critic_only=True)
    burn_in.append(metrics)
  runner.alg.set_critic_only(False)

  thresholds = CandidateGateThresholds()
  accepted_state = runner.snapshot_candidate_state()
  rounds: list[dict[str, Any]] = []
  for round_index in range(1, args.online_rounds + 1):
    before = runner.snapshot_candidate_state()
    old_actor_state = _actor_state(runner.alg.actor)
    obs, update_metrics = _collect_and_update(runner, obs, critic_only=False)
    candidate_actor_state = _actor_state(runner.alg.actor)
    total_kl = _total_actor_kl(runner, base_actor_state)
    precheck_reasons = candidate_precheck(
      update_metrics=update_metrics,
      total_kl_from_base=total_kl,
      parameters_finite=runner.parameters_are_finite(),
      thresholds=thresholds,
    )

    candidate_path = output_dir / f"candidate_round_{round_index:03d}.pt"
    _save_checkpoint(
      runner,
      candidate_path,
      iteration=round_index,
      metadata={
        "accepted": False,
        "stage": "candidate",
        "update_metrics": update_metrics,
        "total_kl_from_base": total_kl,
      },
    )

    old_eval: dict[str, dict[str, Any]] = {}
    candidate_eval: dict[str, dict[str, Any]] = {}
    accepted = False
    reasons = list(precheck_reasons)
    if not reasons:
      old_eval = _evaluate_state(
        runner,
        old_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
      )
      candidate_eval = _evaluate_state(
        runner,
        candidate_actor_state,
        domains=("D0", args.train_domain, args.neighbor_domain),
        num_envs=args.eval_num_envs,
        num_episodes=args.eval_num_episodes,
        seed=args.seed,
        device=args.gate_device,
        repeats=args.gate_repeats,
      )
      accepted, reasons = candidate_gate(
        update_metrics=update_metrics,
        old_eval=old_eval,
        candidate_eval=candidate_eval,
        base_d0_success=baseline_eval["D0"]["success_rate"],
        total_kl_from_base=total_kl,
        parameters_finite=runner.parameters_are_finite(),
        thresholds=thresholds,
        target_domain=args.train_domain,
        retention_domain="D0",
        neighbor_domain=args.neighbor_domain,
      )

    if accepted:
      accepted_state = runner.snapshot_candidate_state()
      accepted_path = output_dir / f"accepted_round_{round_index:03d}.pt"
      _save_checkpoint(
        runner,
        accepted_path,
        iteration=round_index,
        metadata={
          "accepted": True,
          "update_metrics": update_metrics,
          "total_kl_from_base": total_kl,
          "old_eval": old_eval,
          "candidate_eval": candidate_eval,
        },
      )
    else:
      runner.restore_candidate_state(before)
      runner.reduce_after_rejection()
      accepted_state = runner.snapshot_candidate_state()

    record = {
      "round": round_index,
      "accepted": accepted,
      "rejection_reasons": reasons,
      "update_metrics": update_metrics,
      "total_kl_from_base": total_kl,
      "old_eval": old_eval,
      "candidate_eval": candidate_eval,
      "candidate_checkpoint": str(candidate_path),
    }
    rounds.append(record)
    (output_dir / "online_rounds.json").write_text(
      json.dumps(rounds, indent=2, sort_keys=True) + "\n"
    )

  runner.restore_candidate_state(accepted_state)
  final_eval = _evaluate_state(
    runner,
    _actor_state(runner.alg.actor),
    domains=("D0", args.train_domain, args.neighbor_domain),
    num_envs=args.eval_num_envs,
    num_episodes=args.eval_num_episodes,
    seed=args.seed,
    device=args.gate_device,
    repeats=args.gate_repeats,
  )
  final_path = output_dir / "accepted_final.pt"
  result = {
    "task": task,
    "train_domain": args.train_domain,
    "neighbor_domain": args.neighbor_domain,
    "seed": args.seed,
    "warm_start": warm_start,
    "base_checkpoint": str(args.base_checkpoint),
    "critic_burn_in": burn_in,
    "baseline_eval": baseline_eval,
    "rounds": rounds,
    "final_eval": final_eval,
    "final_checkpoint": str(final_path),
  }
  _save_checkpoint(
    runner,
    final_path,
    iteration=args.online_rounds,
    metadata=result,
  )
  (output_dir / "online_refinement_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()
