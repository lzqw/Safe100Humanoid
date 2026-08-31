"""Take one robust PPO step from paired shielded/unshielded GAE rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_cbf_teacher_v31 import _configure_algorithm
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)

METHOD_ID = "paired-multi-rollout-cbf-dual-gae-consensus-v46"
HIERARCHICAL_METHOD_ID = "hierarchical-paired-cbf-dual-gae-consensus-v47"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--rollout-steps", type=int, default=512)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=2.5e-5)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument(
    "--gradient-aggregation",
    choices=("coordinate-trimmed", "paired-mean-coordinate-median"),
    default="coordinate-trimmed",
  )
  parser.add_argument("--max-grad-norm", type=float, default=10.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v46 training seeds must be comma-separated integers") from exc
  if len(seeds) < 3 or len(set(seeds)) != len(seeds):
    raise ValueError("v46 requires at least three unique training seeds")
  return seeds


def _normalized_sha(value: str, label: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError(f"{label} must contain 64 hexadecimal digits")
  return normalized


def _collect_rollout(
  runner,
  base_env,
  action_term,
  *,
  seed: int,
  runtime_filter: bool,
) -> tuple[Any, dict[str, Any]]:
  from rsl_rl.utils import check_nan

  runner.alg.clear_cbf_rollout()
  runner.alg.train_mode()
  action_term.set_runtime_filter_mask(
    torch.full(
      (runner.env.num_envs,),
      runtime_filter,
      dtype=torch.bool,
      device=base_env.device,
    )
  )
  _seed_everything(seed)
  base_env.seed(seed)
  obs, _ = runner.env.reset()
  obs = obs.to(runner.device)
  signature = _initial_state_signature(
    obs,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  n = runner.env.num_envs
  episode_returns = torch.zeros(n, device=runner.env.device)
  episode_max_riser = torch.zeros(n, dtype=torch.long, device=runner.env.device)
  completed_returns: list[float] = []
  completed_risers: list[int] = []
  episode_count = success_count = fall_count = timeout_count = 0
  reward_sum = 0.0
  with torch.no_grad():
    for _ in range(runner.cfg["num_steps_per_env"]):
      raw_actions = runner.alg.act(obs)
      next_obs, rewards, dones, extras = runner.env.step(
        raw_actions.to(runner.env.device)
      )
      check_nan(next_obs, rewards, dones)
      extras = dict(extras)
      episode_returns += rewards
      reward_sum += float(rewards.sum())
      riser = extras["online_stair_index"].long()
      episode_max_riser = torch.maximum(episode_max_riser, riser)
      done_mask = dones.bool()
      if bool(done_mask.any()):
        fell = extras["online_fell"].bool()
        timeouts = extras.get(
          "time_outs", torch.zeros_like(done_mask, dtype=torch.bool)
        ).bool()
        success = base_env.termination_manager.get_term("reached_top").bool()
        ids = done_mask.nonzero(as_tuple=False).flatten()
        completed_returns.extend(
          float(episode_returns[index]) for index in ids.tolist()
        )
        completed_risers.extend(
          int(episode_max_riser[index]) for index in ids.tolist()
        )
        episode_count += len(ids)
        success_count += int((done_mask & success).sum())
        fall_count += int((done_mask & fell).sum())
        timeout_count += int((done_mask & timeouts).sum())
        episode_returns[ids] = 0.0
        episode_max_riser[ids] = 0
      obs = next_obs.to(runner.device)
      runner.alg.process_env_step(
        obs,
        rewards.to(runner.device),
        dones.to(runner.device),
        extras,
      )
    teacher_metrics = runner.alg.relabel_teacher_transitions()
    runner.alg.compute_returns(obs)
  batch = runner.alg.capture_rollout_batch()
  advantages = batch.advantages.flatten()
  summary = {
    "seed": seed,
    "runtime_filter": runtime_filter,
    "initial_state_signature": signature,
    "transition_count": n * runner.cfg["num_steps_per_env"],
    "episode_count": episode_count,
    "success_count": success_count,
    "fall_count": fall_count,
    "timeout_count": timeout_count,
    "success_rate": success_count / max(1, episode_count),
    "fall_rate": fall_count / max(1, episode_count),
    "mean_return": (
      sum(completed_returns) / len(completed_returns)
      if completed_returns
      else None
    ),
    "mean_reached_riser": (
      sum(completed_risers) / len(completed_risers)
      if completed_risers
      else None
    ),
    "mean_reward_per_transition": reward_sum
    / (n * runner.cfg["num_steps_per_env"]),
    "advantage_mean": float(advantages.mean()),
    "advantage_std": float(advantages.std()),
    "observed_runtime_filter_fraction": float(
      teacher_metrics["runtime_filter_enabled_fraction"]
    ),
    "cbf_intervention_fraction": float(
      teacher_metrics["cbf_intervention_fraction"]
    ),
  }
  expected_filter_fraction = float(runtime_filter)
  if not math.isclose(
    summary["observed_runtime_filter_fraction"],
    expected_filter_fraction,
    rel_tol=0.0,
    abs_tol=1.0e-8,
  ):
    raise RuntimeError("v46 rollout filter mask differs from its condition")
  runner.alg.clear_captured_rollout()
  return batch, summary


def _policy_metrics(
  actor,
  reference_actor,
  batches: list[Any],
  *,
  labels: list[dict[str, Any]],
  clip_ratio: float,
) -> dict[str, Any]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  per_batch = []
  total_kl = 0.0
  total_mean_shift = 0.0
  total_count = 0
  with torch.inference_mode():
    for batch, label in zip(batches, labels, strict=True):
      observations = batch.observations.flatten(0, 1)
      actions = batch.actions.flatten(0, 1)
      old_log_prob = batch.actions_log_prob.flatten(0, 1).squeeze(-1)
      advantages = batch.advantages.flatten()
      reference_actor(observations, stochastic_output=True)
      reference_params = tuple(reference_actor.output_distribution_params)
      reference_log_prob = reference_actor.get_output_log_prob(actions)
      actor(observations, stochastic_output=True)
      current_params = tuple(actor.output_distribution_params)
      current_log_prob = actor.get_output_log_prob(actions)
      log_ratio = current_log_prob - old_log_prob
      ratio = torch.exp(log_ratio)
      clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
      baseline = advantages.mean()
      clipped_surrogate = torch.minimum(
        advantages * ratio, advantages * clipped
      ).mean()
      kl = diagonal_gaussian_forward_kl(current_params, reference_params)
      mean_shift = torch.linalg.vector_norm(
        current_params[0] - reference_params[0], dim=-1
      )
      count = len(actions)
      total_kl += float(kl.sum())
      total_mean_shift += float(mean_shift.sum())
      total_count += count
      per_batch.append(
        {
          "seed": label["seed"],
          "runtime_filter": label["runtime_filter"],
          "surrogate_gain": float(clipped_surrogate - baseline),
          "reference_forward_kl": float(kl.mean()),
          "reference_mean_shift": float(mean_shift.mean()),
          "ratio_clip_fraction": float(
            ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio))
            .float()
            .mean()
          ),
          "stored_reference_log_prob_max_abs_error": float(
            (reference_log_prob - old_log_prob).abs().max()
          ),
        }
      )
  gains = [item["surrogate_gain"] for item in per_batch]
  on_gains = [
    item["surrogate_gain"] for item in per_batch if item["runtime_filter"]
  ]
  off_gains = [
    item["surrogate_gain"] for item in per_batch if not item["runtime_filter"]
  ]
  return {
    "mean_surrogate_gain": sum(gains) / len(gains),
    "minimum_surrogate_gain": min(gains),
    "positive_batch_count": sum(value > 0.0 for value in gains),
    "batch_count": len(gains),
    "mean_filter_on_surrogate_gain": sum(on_gains) / len(on_gains),
    "mean_filter_off_surrogate_gain": sum(off_gains) / len(off_gains),
    "reference_forward_kl": total_kl / total_count,
    "reference_mean_shift": total_mean_shift / total_count,
    "per_batch": per_batch,
  }


def _robust_ppo_step(
  actor,
  batches: list[Any],
  labels: list[dict[str, Any]],
  *,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  clip_ratio: float,
  gradient_aggregation: str,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  actor.eval()
  reference_actor = copy.deepcopy(actor).to(device).eval()
  for parameter in reference_actor.parameters():
    parameter.requires_grad_(False)
  parameters = tuple(actor.mlp.parameters())
  for parameter in parameters:
    parameter.requires_grad_(True)
  optimizer = torch.optim.SGD(parameters, lr=learning_rate)
  before = _policy_metrics(
    actor,
    reference_actor,
    batches,
    labels=labels,
    clip_ratio=clip_ratio,
  )
  gradients_by_batch: list[tuple[torch.Tensor, ...]] = []
  policy_losses = []
  for batch in batches:
    observations = batch.observations.flatten(0, 1)
    actions = batch.actions.flatten(0, 1)
    old_log_prob = batch.actions_log_prob.flatten(0, 1).squeeze(-1)
    advantages = batch.advantages.flatten().detach()
    with torch.no_grad():
      reference_actor(observations, stochastic_output=True)
      reference_params = tuple(
        value.detach() for value in reference_actor.output_distribution_params
      )
    actor(observations, stochastic_output=True)
    current_params = tuple(actor.output_distribution_params)
    ratio = torch.exp(actor.get_output_log_prob(actions) - old_log_prob)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.minimum(
      advantages * ratio, advantages * clipped
    ).mean()
    moving_kl = diagonal_gaussian_forward_kl(
      current_params, reference_params
    ).mean()
    loss = policy_loss + moving_kl_beta * moving_kl
    gradients = torch.autograd.grad(loss, parameters)
    if not all(bool(torch.isfinite(value).all()) for value in gradients):
      raise RuntimeError("v46 rollout gradient is non-finite")
    gradients_by_batch.append(tuple(value.detach() for value in gradients))
    policy_losses.append(float(policy_loss.detach()))
  flattened = torch.stack(
    [
      torch.cat([value.flatten() for value in gradients])
      for gradients in gradients_by_batch
    ]
  )
  norms = torch.linalg.vector_norm(flattened, dim=1)
  normalized = flattened / norms.clamp_min(1.0e-12).unsqueeze(1)
  cosines = normalized @ normalized.T
  off_diagonal = cosines[
    ~torch.eye(len(batches), dtype=torch.bool, device=cosines.device)
  ]
  paired_gradient_sets: list[tuple[torch.Tensor, ...]] = []
  paired_gradient_labels: list[int] = []
  for seed in sorted({int(label["seed"]) for label in labels}):
    indices = [
      index for index, label in enumerate(labels) if int(label["seed"]) == seed
    ]
    conditions = {bool(labels[index]["runtime_filter"]) for index in indices}
    if len(indices) != 2 or conditions != {False, True}:
      raise RuntimeError("v47 requires exactly one paired on/off batch per seed")
    paired_gradient_sets.append(
      tuple(
        0.5
        * (
          gradients_by_batch[indices[0]][parameter_index]
          + gradients_by_batch[indices[1]][parameter_index]
        )
        for parameter_index in range(len(parameters))
      )
    )
    paired_gradient_labels.append(seed)
  paired_flattened = torch.stack(
    [
      torch.cat([value.flatten() for value in gradients])
      for gradients in paired_gradient_sets
    ]
  )
  paired_norms = torch.linalg.vector_norm(paired_flattened, dim=1)
  paired_normalized = paired_flattened / paired_norms.clamp_min(1.0e-12).unsqueeze(1)
  paired_seed_cosines = paired_normalized @ paired_normalized.T
  paired_seed_off_diagonal = paired_seed_cosines[
    ~torch.eye(
      len(paired_gradient_sets),
      dtype=torch.bool,
      device=paired_seed_cosines.device,
    )
  ]
  for parameter_index, parameter in enumerate(parameters):
    if gradient_aggregation == "paired-mean-coordinate-median":
      stacked = torch.stack(
        [gradients[parameter_index] for gradients in paired_gradient_sets]
      )
      parameter.grad = stacked.median(dim=0).values
    elif gradient_aggregation == "coordinate-trimmed":
      stacked = torch.stack(
        [gradients[parameter_index] for gradients in gradients_by_batch]
      )
      parameter.grad = stacked.sort(dim=0).values[1:-1].mean(dim=0)
    else:
      raise ValueError(f"unknown v46 gradient aggregation {gradient_aggregation!r}")
  aggregate_gradient_norm = torch.nn.utils.clip_grad_norm_(
    parameters, max_grad_norm
  )
  if not bool(torch.isfinite(aggregate_gradient_norm)):
    raise RuntimeError("v46 aggregate gradient is non-finite")
  optimizer.step()
  unprojected = _policy_metrics(
    actor,
    reference_actor,
    batches,
    labels=labels,
    clip_ratio=clip_ratio,
  )
  reference_state = {
    key: value.detach().clone()
    for key, value in reference_actor.mlp.state_dict().items()
  }
  proposal_state = {
    key: value.detach().clone() for key, value in actor.mlp.state_dict().items()
  }

  def load_interpolation(scale: float) -> None:
    actor.mlp.load_state_dict(
      {
        key: reference_state[key]
        + scale * (proposal_state[key] - reference_state[key])
        for key in reference_state
      },
      strict=True,
    )

  interpolation_scale = 1.0
  projection_iterations = 0
  if unprojected["reference_forward_kl"] > max_reference_kl:
    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_interpolation(middle)
      metrics = _policy_metrics(
        actor,
        reference_actor,
        batches,
        labels=labels,
        clip_ratio=clip_ratio,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    interpolation_scale = low
    load_interpolation(interpolation_scale)
  after = _policy_metrics(
    actor,
    reference_actor,
    batches,
    labels=labels,
    clip_ratio=clip_ratio,
  )
  paired_cosines = []
  for first in range(len(labels)):
    for second in range(first + 1, len(labels)):
      if labels[first]["seed"] == labels[second]["seed"]:
        paired_cosines.append(float(cosines[first, second]))
  return {
    "actor_update_scope": "all-mlp-layers",
    "trainable_parameter_count": sum(
      parameter.numel() for parameter in parameters
    ),
    "optimizer": "sgd",
    "optimizer_updates": 1,
    "gradient_aggregation": gradient_aggregation,
    "per_batch_policy_loss_before": policy_losses,
    "per_batch_gradient_norm": norms.tolist(),
    "pairwise_gradient_cosine_min": float(off_diagonal.min()),
    "pairwise_gradient_cosine_mean": float(off_diagonal.mean()),
    "paired_filter_on_off_gradient_cosines": paired_cosines,
    "paired_filter_on_off_gradient_cosine_mean": sum(paired_cosines)
    / len(paired_cosines),
    "paired_seed_gradient_labels": paired_gradient_labels,
    "paired_seed_gradient_norm": paired_norms.tolist(),
    "paired_seed_gradient_cosine_min": float(paired_seed_off_diagonal.min()),
    "paired_seed_gradient_cosine_mean": float(paired_seed_off_diagonal.mean()),
    "aggregate_gradient_norm_pre_clip": float(aggregate_gradient_norm),
    "before": before,
    "unprojected_after": unprojected,
    "after": after,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "parameter_interpolation_scale": interpolation_scale,
      "projection_iterations": projection_iterations,
    },
  }, optimizer


def main() -> None:
  args = _parse_args()
  training_seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or args.rollout_steps < 64:
    raise ValueError("v46 rollout dimensions are too small")
  if not 1.0e-5 <= args.actor_learning_rate <= 1.0e-2:
    raise ValueError("v46 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v46 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.001:
    raise ValueError("v46 reference-KL cap must lie in (0, 0.001]")
  if not 0.0 < args.clip_ratio <= 0.5 or args.max_grad_norm <= 0.0:
    raise ValueError("v46 clip ratio and gradient bound differ")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v46 requires a clean committed worktree")
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(
    args.expected_base_sha256, "v46 expected base SHA-256"
  ):
    raise RuntimeError("v46 base checkpoint SHA-256 differs")
  if output.exists():
    raise FileExistsError(output)
  method_id = (
    HIERARCHICAL_METHOD_ID
    if args.gradient_aggregation == "paired-mean-coordinate-median"
    else METHOD_ID
  )
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": training_seeds,
      "rollout_conditions": ["filter_on", "filter_off"],
      "optimization_seed": args.optimization_seed,
      "gradient_aggregation": args.gradient_aggregation,
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
  from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner

  _seed_everything(training_seeds[0])
  env_cfg = load_env_cfg(TASK_ID, play=True)
  shift = configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=True,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  reward = configure_paper_dual_reward(
    env_cfg, "raw_moderate", runtime_filter_during_training=True
  )
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = training_seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  agent_cfg.seed = training_seeds[0]
  agent_cfg.num_steps_per_env = args.rollout_steps
  _configure_algorithm(agent_cfg, "A0", preflight=False)
  agent_cfg.algorithm.minimum_std = 0.05
  agent_cfg.algorithm.maximum_std = 0.05
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner = CbfTeacherV30Runner(
    env, asdict(agent_cfg), log_dir=None, device=args.device
  )
  action_term = base_env.action_manager.get_term("joint_pos")
  try:
    warm_start = runner.load_initial_checkpoint(
      str(checkpoint), map_location=args.device
    )
    initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
    runner.alg.freeze_round_reference()
    batches = []
    rollout_summaries = []
    signatures: dict[int, str] = {}
    for seed in training_seeds:
      for runtime_filter in (True, False):
        batch, rollout = _collect_rollout(
          runner,
          base_env,
          action_term,
          seed=seed,
          runtime_filter=runtime_filter,
        )
        previous_signature = signatures.setdefault(
          seed, rollout["initial_state_signature"]
        )
        if previous_signature != rollout["initial_state_signature"]:
          raise RuntimeError("v46 paired initial states differ")
        batches.append(batch)
        rollout_summaries.append(rollout)
        print(json.dumps({"rollout_completed": rollout}, sort_keys=True), flush=True)
    _seed_everything(args.optimization_seed)
    training, optimizer = _robust_ppo_step(
      runner.alg.actor,
      batches,
      rollout_summaries,
      learning_rate=args.actor_learning_rate,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      clip_ratio=args.clip_ratio,
      gradient_aggregation=args.gradient_aggregation,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    after = training["after"]
    offline_gate_passed = (
      after["positive_batch_count"] == after["batch_count"]
      and after["mean_filter_on_surrogate_gain"] > 0.0
      and after["mean_filter_off_surrogate_gain"] > 0.0
      and after["reference_forward_kl"] <= args.max_reference_kl
    )
    final_actor_state = actor_state(runner.alg.actor)
    final_actor_hash = actor_state_sha256(final_actor_state)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload["actor_state_dict"] = {
      key: value.cpu() for key, value in final_actor_state.items()
    }
    source_payload["multi_rollout_gae_optimizer_state_dict"] = (
      optimizer.state_dict()
    )
    source_payload["iter"] = int(source_payload.get("iter", 0)) + 1
    infos = dict(source_payload.get("infos") or {})
    info_key = (
      "hierarchical_paired_gae_consensus_v47"
      if method_id == HIERARCHICAL_METHOD_ID
      else "multi_rollout_gae_consensus_v46"
    )
    infos[info_key] = {
      "method_id": method_id,
      "source_git_commit": _git(repo, "rev-parse", "HEAD"),
      "training_seeds": training_seeds,
      "optimization_seed": args.optimization_seed,
      "rollout_conditions": ["filter_on", "filter_off"],
      "gradient_aggregation": args.gradient_aggregation,
      "offline_gate_passed": offline_gate_passed,
    }
    source_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, source_payload)
    summary = {
      "schema_version": 1,
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "training_seeds": training_seeds,
      "num_envs": args.num_envs,
      "rollout_steps": args.rollout_steps,
      "rollout_batch_count": len(batches),
      "rollout_summaries": rollout_summaries,
      "shift": shift,
      "reward": reward,
      "warm_start": warm_start,
      "optimization_seed": args.optimization_seed,
      "base_checkpoint_sha256": checkpoint_sha,
      "initial_actor_sha256": initial_actor_hash,
      "candidate_path": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "candidate_actor_sha256": final_actor_hash,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "clip_ratio": args.clip_ratio,
      "max_grad_norm": args.max_grad_norm,
      "gradient_aggregation": args.gradient_aggregation,
      "offline_gate_passed": offline_gate_passed,
      "training": training,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
