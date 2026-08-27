"""Take one robust PPO step from successful and failed filter-free episodes."""

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
from elite_self_imitation_v41 import METHOD_ID as SOURCE_METHOD_ID
from elite_self_imitation_v41 import _elite_metrics
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _dataset_reference_metrics,
  _git,
  _seed_everything,
)

METHOD_ID = "episode-balanced-filter-free-outcome-ppo-v44"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, action="append", required=True)
  parser.add_argument("--expected-dataset-sha256", action="append", required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=1.0e-4)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument(
    "--gradient-aggregation",
    choices=("mean", "coordinate-trimmed"),
    default="coordinate-trimmed",
  )
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _normalized_sha(value: str, label: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError(f"{label} must contain 64 hexadecimal digits")
  return normalized


def _episode_balanced_dataset(
  payloads: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[int]]:
  """Build zero-mean, episode-balanced outcome advantages for every seed."""
  observations: list[torch.Tensor] = []
  actions: list[torch.Tensor] = []
  elites: list[torch.Tensor] = []
  weights: list[torch.Tensor] = []
  seed_ids: list[torch.Tensor] = []
  seed_summaries: list[dict[str, Any]] = []
  source_training_seeds: list[int] = []
  seen_seeds: set[int] = set()
  global_seed_id = 0
  for payload_index, payload in enumerate(payloads):
    local_observations = payload["observations"]
    local_actions = payload["sampled_actions"]
    local_elite = payload["elite"].bool()
    environment_ids = payload["environment_ids"].long()
    if (
      local_observations.ndim != 2
      or local_actions.ndim != 2
      or local_elite.ndim != 1
      or environment_ids.ndim != 1
      or not (
        len(local_observations)
        == len(local_actions)
        == len(local_elite)
        == len(environment_ids)
      )
    ):
      raise RuntimeError(f"v44 dataset {payload_index} tensor shapes differ")
    if observations and (
      local_observations.shape[1] != observations[0].shape[1]
      or local_actions.shape[1] != actions[0].shape[1]
    ):
      raise RuntimeError("v44 dataset feature shapes differ")
    unique_ids = torch.unique(environment_ids, sorted=True)
    if (
      unique_ids.numel() < 2
      or int(unique_ids[0]) != 0
      or not torch.equal(unique_ids, torch.arange(len(unique_ids)))
    ):
      raise RuntimeError(f"v44 dataset {payload_index} environment IDs are not dense")
    training_seeds = [int(seed) for seed in payload["training_seeds"]]
    duplicate_seeds = sorted(seen_seeds.intersection(training_seeds))
    if duplicate_seeds:
      raise RuntimeError(f"v44 datasets repeat training seeds {duplicate_seeds}")
    if len(unique_ids) % len(training_seeds):
      raise RuntimeError(f"v44 dataset {payload_index} seed groups are uneven")
    seen_seeds.update(training_seeds)
    source_training_seeds.extend(training_seeds)
    envs_per_seed = len(unique_ids) // len(training_seeds)
    episode_lengths = torch.bincount(environment_ids, minlength=len(unique_ids))
    elite_counts = torch.bincount(
      environment_ids, weights=local_elite.float(), minlength=len(unique_ids)
    )
    if bool(((elite_counts != 0) & (elite_counts != episode_lengths)).any()):
      raise RuntimeError(f"v44 dataset {payload_index} has mixed episode outcomes")
    episode_success = elite_counts == episode_lengths
    local_weights = torch.empty(len(local_elite), dtype=torch.float32)
    local_seed_ids = torch.empty(len(local_elite), dtype=torch.long)
    for seed_offset, training_seed in enumerate(training_seeds):
      first_env = seed_offset * envs_per_seed
      last_env = first_env + envs_per_seed
      success = episode_success[first_env:last_env]
      success_count = int(success.sum())
      failure_count = envs_per_seed - success_count
      if success_count < 1 or failure_count < 1:
        raise RuntimeError(
          f"v44 training seed {training_seed} needs successes and failures"
        )
      env_mask = (environment_ids >= first_env) & (environment_ids < last_env)
      transition_success = episode_success[environment_ids[env_mask]]
      episode_weight = torch.where(
        transition_success,
        torch.full_like(transition_success, 0.5 / success_count, dtype=torch.float32),
        torch.full_like(transition_success, -0.5 / failure_count, dtype=torch.float32),
      )
      local_weights[env_mask] = episode_weight / episode_lengths[
        environment_ids[env_mask]
      ].float()
      local_seed_ids[env_mask] = global_seed_id
      seed_summaries.append(
        {
          "seed_id": global_seed_id,
          "training_seed": training_seed,
          "episode_count": envs_per_seed,
          "success_count": success_count,
          "failure_count": failure_count,
          "transition_count": int(env_mask.sum()),
        }
      )
      global_seed_id += 1
    observations.append(local_observations)
    actions.append(local_actions)
    elites.append(local_elite)
    weights.append(local_weights)
    seed_ids.append(local_seed_ids)
  dataset = {
    "observations": torch.cat(observations),
    "sampled_actions": torch.cat(actions),
    "elite": torch.cat(elites),
    "weights": torch.cat(weights),
    "seed_ids": torch.cat(seed_ids),
  }
  return dataset, seed_summaries, source_training_seeds


def _outcome_metrics(
  actor,
  reference_actor,
  dataset: dict[str, torch.Tensor],
  *,
  seed_count: int,
  clip_ratio: float,
  device: str,
  batch_size: int,
) -> dict[str, Any]:
  surrogate = torch.zeros(seed_count, dtype=torch.float64)
  clipped_surrogate = torch.zeros_like(surrogate)
  baseline = torch.zeros_like(surrogate)
  success_log_ratio = 0.0
  success_weight = 0.0
  failure_log_ratio = 0.0
  failure_weight = 0.0
  clipped_count = 0
  count = 0
  with torch.inference_mode():
    for start in range(0, len(dataset["observations"]), batch_size):
      stop = min(len(dataset["observations"]), start + batch_size)
      obs = dataset["observations"][start:stop].to(device)
      actions = dataset["sampled_actions"][start:stop].to(device)
      signed_weights = dataset["weights"][start:stop].to(device)
      ids = dataset["seed_ids"][start:stop]
      actor_obs = {"actor": obs}
      reference_actor(actor_obs, stochastic_output=True)
      reference_log_prob = reference_actor.get_output_log_prob(actions)
      actor(actor_obs, stochastic_output=True)
      current_log_prob = actor.get_output_log_prob(actions)
      log_ratio = current_log_prob - reference_log_prob
      ratio = torch.exp(log_ratio)
      clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
      contribution = signed_weights * ratio
      clipped_contribution = torch.minimum(
        contribution, signed_weights * clipped
      )
      for seed_id in range(seed_count):
        mask = ids == seed_id
        if bool(mask.any()):
          surrogate[seed_id] += float(contribution[mask.to(device)].sum())
          clipped_surrogate[seed_id] += float(
            clipped_contribution[mask.to(device)].sum()
          )
          baseline[seed_id] += float(signed_weights[mask.to(device)].sum())
      positive = signed_weights > 0
      negative = signed_weights < 0
      success_log_ratio += float((signed_weights[positive] * log_ratio[positive]).sum())
      success_weight += float(signed_weights[positive].sum())
      failure_log_ratio += float(
        ((-signed_weights[negative]) * log_ratio[negative]).sum()
      )
      failure_weight += float((-signed_weights[negative]).sum())
      clipped_count += int(((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).sum())
      count += len(obs)
  gains = surrogate - baseline
  clipped_gains = clipped_surrogate - baseline
  return {
    "outcome_surrogate_gain": float(gains.mean()),
    "clipped_outcome_surrogate_gain": float(clipped_gains.mean()),
    "minimum_seed_surrogate_gain": float(gains.min()),
    "median_seed_surrogate_gain": float(gains.median()),
    "positive_seed_count": int((gains > 0).sum()),
    "per_seed_surrogate_gain": gains.tolist(),
    "per_seed_clipped_surrogate_gain": clipped_gains.tolist(),
    "episode_balanced_success_log_ratio": success_log_ratio / success_weight,
    "episode_balanced_failure_log_ratio": failure_log_ratio / failure_weight,
    "transition_clip_fraction": clipped_count / count,
  }


def _distill_outcome_ppo(
  actor,
  dataset: dict[str, torch.Tensor],
  *,
  seed_count: int,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  clip_ratio: float,
  gradient_aggregation: str,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  dataset = {key: value.clone() for key, value in dataset.items()}
  actor.eval()
  reference_actor = copy.deepcopy(actor).to(device).eval()
  for parameter in reference_actor.parameters():
    parameter.requires_grad_(False)
  linear_layers = [
    module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
  ]
  if not linear_layers:
    raise RuntimeError("v44 actor MLP has no linear output layer")
  for parameter in actor.mlp.parameters():
    parameter.requires_grad_(False)
  parameters = list(linear_layers[-1].parameters())
  for parameter in parameters:
    parameter.requires_grad_(True)
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  batch_size = math.ceil(len(dataset["observations"]) / seed_count)
  before = _outcome_metrics(
    actor,
    reference_actor,
    dataset,
    seed_count=seed_count,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  elite_before = _elite_metrics(
    actor,
    reference_actor,
    dataset["observations"],
    dataset["sampled_actions"],
    dataset["elite"],
    device=device,
    batch_size=batch_size,
  )
  trust_before = _dataset_reference_metrics(
    actor,
    reference_actor,
    dataset["observations"],
    device=device,
    batch_size=batch_size,
  )
  gradients_by_seed: list[tuple[torch.Tensor, ...]] = []
  losses_by_seed: list[float] = []
  for seed_id in range(seed_count):
    indices = (dataset["seed_ids"] == seed_id).nonzero(as_tuple=False).flatten()
    obs = dataset["observations"][indices].to(device)
    actions = dataset["sampled_actions"][indices].to(device)
    signed_weights = dataset["weights"][indices].to(device)
    actor_obs = {"actor": obs}
    with torch.no_grad():
      reference_actor(actor_obs, stochastic_output=True)
      reference_params = tuple(
        value.detach() for value in reference_actor.output_distribution_params
      )
      reference_log_prob = reference_actor.get_output_log_prob(actions).detach()
    actor(actor_obs, stochastic_output=True)
    current_params = tuple(actor.output_distribution_params)
    current_log_prob = actor.get_output_log_prob(actions)
    ratio = torch.exp(current_log_prob - reference_log_prob)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(signed_weights * ratio, signed_weights * clipped)
    moving_kl = diagonal_gaussian_forward_kl(
      current_params, reference_params
    ).mean()
    loss = -surrogate.sum() + moving_kl_beta * moving_kl
    gradients = torch.autograd.grad(loss, parameters)
    if not all(bool(torch.isfinite(value).all()) for value in gradients):
      raise RuntimeError(f"v44 seed {seed_id} gradient is non-finite")
    gradients_by_seed.append(tuple(value.detach() for value in gradients))
    losses_by_seed.append(float(loss.detach()))
  flattened = torch.stack(
    [torch.cat([value.flatten() for value in gradients]) for gradients in gradients_by_seed]
  )
  norms = torch.linalg.vector_norm(flattened, dim=1)
  normalized = flattened / norms.clamp_min(1.0e-12).unsqueeze(1)
  cosines = normalized @ normalized.T
  off_diagonal = cosines[~torch.eye(seed_count, dtype=torch.bool, device=device)]
  for parameter_index, parameter in enumerate(parameters):
    stacked = torch.stack(
      [gradients[parameter_index] for gradients in gradients_by_seed]
    )
    if gradient_aggregation == "coordinate-trimmed":
      if seed_count < 3:
        raise RuntimeError("v44 trimmed aggregation requires at least three seeds")
      aggregate = stacked.sort(dim=0).values[1:-1].mean(dim=0)
    else:
      aggregate = stacked.mean(dim=0)
    parameter.grad = aggregate
  gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
  if not bool(torch.isfinite(gradient_norm)):
    raise RuntimeError("v44 aggregate actor gradient is non-finite")
  optimizer.step()
  unprojected_outcome = _outcome_metrics(
    actor,
    reference_actor,
    dataset,
    seed_count=seed_count,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  unprojected_trust = _dataset_reference_metrics(
    actor,
    reference_actor,
    dataset["observations"],
    device=device,
    batch_size=batch_size,
  )
  reference_state = {
    key: value.detach().clone() for key, value in reference_actor.mlp.state_dict().items()
  }
  proposal_state = {
    key: value.detach().clone() for key, value in actor.mlp.state_dict().items()
  }

  def load_interpolation(scale: float) -> None:
    actor.mlp.load_state_dict(
      {
        key: reference_state[key] + scale * (proposal_state[key] - reference_state[key])
        for key in reference_state
      },
      strict=True,
    )

  interpolation_scale = 1.0
  projection_iterations = 0
  if unprojected_trust["reference_forward_kl"] > max_reference_kl:
    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_interpolation(middle)
      metrics = _dataset_reference_metrics(
        actor,
        reference_actor,
        dataset["observations"],
        device=device,
        batch_size=batch_size,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    interpolation_scale = low
    load_interpolation(interpolation_scale)
  after = _outcome_metrics(
    actor,
    reference_actor,
    dataset,
    seed_count=seed_count,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  elite_after = _elite_metrics(
    actor,
    reference_actor,
    dataset["observations"],
    dataset["sampled_actions"],
    dataset["elite"],
    device=device,
    batch_size=batch_size,
  )
  trust_after = _dataset_reference_metrics(
    actor,
    reference_actor,
    dataset["observations"],
    device=device,
    batch_size=batch_size,
  )
  return {
    "dataset_transition_count": len(dataset["observations"]),
    "elite_transition_count": int(dataset["elite"].sum()),
    "actor_update_scope": "last-layer",
    "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
    "optimizer_updates": 1,
    "gradient_aggregation": gradient_aggregation,
    "per_seed_policy_loss_before": losses_by_seed,
    "per_seed_gradient_norm": norms.tolist(),
    "pairwise_gradient_cosine_min": float(off_diagonal.min()),
    "pairwise_gradient_cosine_mean": float(off_diagonal.mean()),
    "aggregate_gradient_norm_pre_clip": float(gradient_norm),
    "before": before,
    "unprojected_after": unprojected_outcome,
    "after": after,
    "elite_before": elite_before,
    "elite_after": elite_after,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "before": trust_before,
      "unprojected_after": unprojected_trust,
      "after": trust_after,
      "parameter_interpolation_scale": interpolation_scale,
      "projection_iterations": projection_iterations,
    },
  }, optimizer


def main() -> None:
  args = _parse_args()
  if len(args.dataset) != len(args.expected_dataset_sha256):
    raise ValueError("v44 requires one expected SHA-256 per dataset")
  if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-4:
    raise ValueError("v44 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v44 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.01:
    raise ValueError("v44 reference-KL cap must lie in (0, 0.01]")
  if not 0.0 < args.clip_ratio <= 0.5 or args.max_grad_norm <= 0.0:
    raise ValueError("v44 clip ratio and gradient norm differ")
  repo = args.repo.resolve()
  dataset_paths = [path.resolve() for path in args.dataset]
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v44 requires a clean committed worktree")
  if not all(path.is_file() for path in dataset_paths) or not checkpoint.is_file():
    raise FileNotFoundError("v44 dataset or checkpoint is missing")
  if output.exists():
    raise FileExistsError(output)
  dataset_shas = [file_sha256(path) for path in dataset_paths]
  for index, (actual, expected) in enumerate(
    zip(dataset_shas, args.expected_dataset_sha256, strict=True)
  ):
    if actual != _normalized_sha(expected, f"v44 expected dataset {index} SHA-256"):
      raise RuntimeError(f"v44 elite dataset {index} SHA-256 differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(
    args.expected_base_sha256, "v44 expected checkpoint SHA-256"
  ):
    raise RuntimeError("v44 base checkpoint SHA-256 differs")
  required = {
    "method_id",
    "training_seeds",
    "actor_sha256",
    "observations",
    "sampled_actions",
    "environment_ids",
    "elite",
  }
  payloads = []
  actor_hash: str | None = None
  for index, path in enumerate(dataset_paths):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted(required - set(payload))
    if missing:
      raise RuntimeError(f"v44 dataset {index} is missing {missing}")
    if payload["method_id"] != SOURCE_METHOD_ID:
      raise RuntimeError(f"v44 dataset {index} source method differs")
    if actor_hash is None:
      actor_hash = payload["actor_sha256"]
    elif actor_hash != payload["actor_sha256"]:
      raise RuntimeError("v44 source datasets use different actors")
    payloads.append(payload)
  dataset, seed_summaries, source_training_seeds = _episode_balanced_dataset(payloads)
  seed_count = len(seed_summaries)
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "source_dataset_sha256s": dataset_shas,
      "base_checkpoint_sha256": checkpoint_sha,
      "optimization_seed": args.optimization_seed,
      "gradient_aggregation": args.gradient_aggregation,
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context

  _seed_everything(args.optimization_seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=False,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  env_cfg.scene.num_envs = 2
  env_cfg.seed = args.optimization_seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  agent_cfg = load_rl_cfg(TASK_ID)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
      raise RuntimeError("v44 task has no runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
    if initial_actor_hash != actor_hash:
      raise RuntimeError("v44 dataset actor differs from the base actor")
    _seed_everything(args.optimization_seed)
    training, optimizer = _distill_outcome_ppo(
      runner.alg.actor,
      dataset,
      seed_count=seed_count,
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
      after["clipped_outcome_surrogate_gain"] > 0.0
      and after["positive_seed_count"] >= math.ceil(seed_count / 2)
      and training["trust_region"]["after"]["reference_forward_kl"]
      <= args.max_reference_kl
    )
    final_actor_state = actor_state(runner.alg.actor)
    final_actor_hash = actor_state_sha256(final_actor_state)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_payload["actor_state_dict"] = {
      key: value.cpu() for key, value in final_actor_state.items()
    }
    source_payload["outcome_ppo_optimizer_state_dict"] = optimizer.state_dict()
    source_payload["iter"] = int(source_payload.get("iter", 0)) + 1
    infos = dict(source_payload.get("infos") or {})
    infos["episode_balanced_outcome_ppo_v44"] = {
      "method_id": METHOD_ID,
      "source_git_commit": _git(repo, "rev-parse", "HEAD"),
      "source_dataset_sha256s": dataset_shas,
      "source_training_seeds": source_training_seeds,
      "optimization_seed": args.optimization_seed,
      "gradient_aggregation": args.gradient_aggregation,
      "offline_gate_passed": offline_gate_passed,
    }
    source_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, source_payload)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "source_datasets": [str(path) for path in dataset_paths],
      "source_dataset_sha256s": dataset_shas,
      "source_training_seeds": source_training_seeds,
      "seed_summaries": seed_summaries,
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
