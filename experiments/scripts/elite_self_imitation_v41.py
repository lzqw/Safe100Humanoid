"""Fit the deterministic actor to successful filter-free stochastic rollouts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from cbf_teacher_v31_protocol import CONTEXTS
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _dataset_reference_metrics,
  _first_episode_rollout,
  _git,
  _seed_everything,
)

METHOD_ID = "filter-free-elite-self-imitation-v41"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=1.0e-4)
  parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
  parser.add_argument("--minibatches", type=int, default=4)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v41 training seeds must be comma-separated integers") from exc
  if not seeds or len(set(seeds)) != len(seeds):
    raise ValueError("v41 requires one or more unique training seeds")
  return seeds


def _elite_metrics(
  actor,
  reference_actor,
  observations: torch.Tensor,
  target_actions: torch.Tensor,
  elite: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  distance_sum = 0.0
  loss_sum = 0.0
  reference_target_distance_sum = 0.0
  elite_count = 0
  with torch.inference_mode():
    for start in range(0, len(observations), batch_size):
      stop = min(len(observations), start + batch_size)
      obs = observations[start:stop].to(device)
      targets = target_actions[start:stop].to(device)
      mask = elite[start:stop].to(device)
      actor_obs = {"actor": obs}
      reference_actor(actor_obs, stochastic_output=True)
      reference_mean = reference_actor.output_distribution_params[0]
      actor(actor_obs, stochastic_output=True)
      current_mean = actor.output_distribution_params[0]
      selected = mask.to(current_mean.dtype)
      distance_sum += float(
        (
          selected
          * torch.linalg.vector_norm(current_mean - targets, dim=-1)
        ).sum()
      )
      reference_target_distance_sum += float(
        (
          selected
          * torch.linalg.vector_norm(reference_mean - targets, dim=-1)
        ).sum()
      )
      per_transition = F.smooth_l1_loss(
        current_mean, targets, reduction="none", beta=0.05
      ).mean(dim=-1)
      loss_sum += float((selected * per_transition).sum())
      elite_count += int(mask.sum())
  if elite_count < 1:
    raise RuntimeError("v41 elite dataset contains no successful transitions")
  return {
    "elite_target_distance": distance_sum / elite_count,
    "reference_elite_target_distance": (
      reference_target_distance_sum / elite_count
    ),
    "elite_smooth_l1": loss_sum / elite_count,
  }


def _distill_elite(
  actor,
  observations: torch.Tensor,
  target_actions: torch.Tensor,
  elite: torch.Tensor,
  *,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  epochs: int,
  minibatches: int,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  # Rollout tensors were created under inference mode. Cloning here produces
  # ordinary tensors that autograd may safely retain for the supervised loss.
  observations = observations.clone()
  target_actions = target_actions.clone()
  elite = elite.clone()
  actor.eval()
  reference_actor = copy.deepcopy(actor).to(device).eval()
  for parameter in reference_actor.parameters():
    parameter.requires_grad_(False)
  linear_layers = [
    module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
  ]
  if not linear_layers:
    raise RuntimeError("v41 actor MLP has no linear output layer")
  for parameter in actor.mlp.parameters():
    parameter.requires_grad_(False)
  parameters = list(linear_layers[-1].parameters())
  for parameter in parameters:
    parameter.requires_grad_(True)
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  total = len(observations)
  batch_size = math.ceil(total / minibatches)
  before = _elite_metrics(
    actor,
    reference_actor,
    observations,
    target_actions,
    elite,
    device=device,
    batch_size=batch_size,
  )
  trust_before = _dataset_reference_metrics(
    actor,
    reference_actor,
    observations,
    device=device,
    batch_size=batch_size,
  )
  updates = 0
  clipped = 0
  gradient_norm_max = 0.0
  elite_loss_sum = 0.0
  moving_kl_sum = 0.0
  minibatches_with_elite = 0
  for _ in range(epochs):
    permutation = torch.randperm(total)
    for start in range(0, total, batch_size):
      indices = permutation[start : start + batch_size]
      obs = observations[indices].to(device)
      targets = target_actions[indices].to(device)
      mask = elite[indices].to(device)
      actor_obs = {"actor": obs}
      with torch.no_grad():
        reference_actor(actor_obs, stochastic_output=True)
        reference_params = tuple(
          value.detach() for value in reference_actor.output_distribution_params
        )
      actor(actor_obs, stochastic_output=True)
      current_params = tuple(actor.output_distribution_params)
      per_transition = F.smooth_l1_loss(
        current_params[0], targets, reduction="none", beta=0.05
      ).mean(dim=-1)
      if bool(mask.any()):
        elite_loss = per_transition[mask].mean()
      else:
        elite_loss = current_params[0].sum() * 0.0
      moving_kl = diagonal_gaussian_forward_kl(
        current_params, reference_params
      ).mean()
      loss = elite_loss + moving_kl_beta * moving_kl
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v41 actor gradient is non-finite")
      clipped += int(float(gradient_norm) > max_grad_norm)
      gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
      optimizer.step()
      updates += 1
      minibatches_with_elite += int(bool(mask.any()))
      elite_loss_sum += float(elite_loss.detach())
      moving_kl_sum += float(moving_kl.detach())
  unprojected_after = _elite_metrics(
    actor,
    reference_actor,
    observations,
    target_actions,
    elite,
    device=device,
    batch_size=batch_size,
  )
  unprojected_trust = _dataset_reference_metrics(
    actor,
    reference_actor,
    observations,
    device=device,
    batch_size=batch_size,
  )
  interpolation_scale = 1.0
  projection_iterations = 0
  if unprojected_trust["reference_forward_kl"] > max_reference_kl:
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

    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_interpolation(middle)
      metrics = _dataset_reference_metrics(
        actor,
        reference_actor,
        observations,
        device=device,
        batch_size=batch_size,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    interpolation_scale = low
    load_interpolation(interpolation_scale)
  after = _elite_metrics(
    actor,
    reference_actor,
    observations,
    target_actions,
    elite,
    device=device,
    batch_size=batch_size,
  )
  trust_after = _dataset_reference_metrics(
    actor,
    reference_actor,
    observations,
    device=device,
    batch_size=batch_size,
  )
  return {
    "dataset_transition_count": total,
    "elite_transition_count": int(elite.sum()),
    "elite_transition_fraction": float(elite.float().mean()),
    "actor_update_scope": "last-layer",
    "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
    "optimizer_updates": updates,
    "minibatches_with_elite": minibatches_with_elite,
    "actor_gradient_clipped_fraction": clipped / max(1, updates),
    "actor_gradient_norm_pre_clip_max": gradient_norm_max,
    "elite_loss_during_update": elite_loss_sum / max(1, updates),
    "moving_kl_during_update": moving_kl_sum / max(1, updates),
    "before": before,
    "unprojected_after": unprojected_after,
    "after": after,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "before": trust_before,
      "unprojected_after": unprojected_trust,
      "after": trust_after,
      "parameter_interpolation_scale": interpolation_scale,
      "projection_iterations": projection_iterations,
    },
  }, optimizer


def _write_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  training_seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or args.minibatches < 1 or args.max_grad_norm <= 0.0:
    raise ValueError("v41 environment/minibatch/gradient values must be positive")
  if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-4:
    raise ValueError("v41 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v41 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.01:
    raise ValueError("v41 reference-KL cap must lie in (0, 0.01]")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v41 requires a clean committed worktree")
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != args.expected_base_sha256.strip().lower():
    raise RuntimeError("v41 base checkpoint SHA-256 differs")
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "training_seeds": training_seeds,
      "base_checkpoint_sha256": checkpoint_sha,
    },
  )
  sys.path.insert(0, str(repo))
  dataset_chunks: list[dict[str, torch.Tensor]] = []
  elite_id_chunks: list[torch.Tensor] = []
  outcomes: list[dict[str, Any]] = []
  per_seed: list[dict[str, Any]] = []
  signatures: dict[str, str] = {}
  actor_hash: str | None = None
  runner = None
  exploration_std: list[float] | None = None
  for seed_index, seed in enumerate(training_seeds):
    result, runner, env = _first_episode_rollout(
      repo=repo,
      checkpoint=checkpoint,
      context=args.context,
      seed=seed,
      num_envs=args.num_envs,
      runtime_filter=False,
      device=args.device,
      retain_runner=True,
      stochastic_policy=True,
    )
    env.close()
    if actor_hash is None:
      actor_hash = result["actor_sha256"]
      distribution = runner.alg.actor.distribution
      if not hasattr(distribution, "std_param"):
        raise RuntimeError("v41 requires a direct Gaussian std parameter")
      exploration_std = distribution.std_param.detach().cpu().tolist()
    elif actor_hash != result["actor_sha256"]:
      raise RuntimeError("v41 training seeds loaded different actors")
    signatures[str(seed)] = result["initial_state_signature"]
    local_elite = result["success"].nonzero(as_tuple=False).flatten()
    offset = seed_index * args.num_envs
    elite_id_chunks.append(local_elite + offset)
    chunk = {key: value for key, value in result["dataset"].items()}
    chunk["environment_ids"] = chunk["environment_ids"] + offset
    dataset_chunks.append(chunk)
    per_seed.append(
      {
        "seed": seed,
        "initial_state_signature": result["initial_state_signature"],
        "success_count": int(result["success"].sum()),
        "fall_count": int(result["fell"].sum()),
      }
    )
    for env_id in range(args.num_envs):
      outcomes.append(
        {
          "training_seed": seed,
          "environment_id": env_id,
          "global_environment_id": offset + env_id,
          "success": bool(result["success"][env_id]),
          "fell": bool(result["fell"][env_id]),
          "steps": int(result["steps"][env_id]),
        }
      )
  if runner is None or actor_hash is None or exploration_std is None:
    raise RuntimeError("v41 did not construct a training actor")
  dataset = {
    key: torch.cat([chunk[key] for chunk in dataset_chunks])
    for key in dataset_chunks[0]
  }
  elite_environment_ids = torch.cat(elite_id_chunks)
  elite = torch.isin(dataset["environment_ids"], elite_environment_ids)
  if not bool(elite.any()):
    raise RuntimeError("v41 collected no successful filter-free transitions")
  dataset_path = output / "elite_dataset.pt"
  _atomic_torch(
    dataset_path,
    {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "training_seeds": training_seeds,
      "initial_state_signatures": signatures,
      "actor_sha256": actor_hash,
      "exploration_std": exploration_std,
      "elite_environment_ids": elite_environment_ids,
      "observations": dataset["observations"],
      "sampled_actions": dataset["nominal_actions"],
      "environment_ids": dataset["environment_ids"],
      "elite": elite,
    },
  )
  _write_outcomes(output / "rollout_outcomes.csv", outcomes)
  _seed_everything(args.optimization_seed)
  training, optimizer = _distill_elite(
    runner.alg.actor,
    dataset["observations"],
    dataset["nominal_actions"],
    elite,
    learning_rate=args.actor_learning_rate,
    moving_kl_beta=args.moving_kl_beta,
    max_reference_kl=args.max_reference_kl,
    epochs=args.epochs,
    minibatches=args.minibatches,
    max_grad_norm=args.max_grad_norm,
    device=args.device,
  )
  final_actor_state = actor_state(runner.alg.actor)
  final_actor_hash = actor_state_sha256(final_actor_state)
  source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  source_payload["actor_state_dict"] = {
    key: value.cpu() for key, value in final_actor_state.items()
  }
  source_payload["elite_optimizer_state_dict"] = optimizer.state_dict()
  source_payload["iter"] = int(source_payload.get("iter", 0)) + args.epochs
  infos = dict(source_payload.get("infos") or {})
  infos["elite_self_imitation_v41"] = {
    "method_id": METHOD_ID,
    "source_git_commit": _git(repo, "rev-parse", "HEAD"),
    "training_seeds": training_seeds,
    "optimization_seed": args.optimization_seed,
    "actor_update_scope": "last-layer",
    "max_reference_kl": args.max_reference_kl,
  }
  source_payload["infos"] = infos
  candidate_path = output / "candidate.pt"
  _atomic_torch(candidate_path, source_payload)
  summary = {
    "schema_version": 1,
    "method_id": METHOD_ID,
    "git_commit": _git(repo, "rev-parse", "HEAD"),
    "context": args.context,
    "training_seeds": training_seeds,
    "optimization_seed": args.optimization_seed,
    "num_envs_per_seed": args.num_envs,
    "total_initial_episodes": args.num_envs * len(training_seeds),
    "success_count": sum(item["success_count"] for item in per_seed),
    "fall_count": sum(item["fall_count"] for item in per_seed),
    "per_seed": per_seed,
    "initial_state_signatures": signatures,
    "exploration_std": exploration_std,
    "base_checkpoint_sha256": checkpoint_sha,
    "initial_actor_sha256": actor_hash,
    "dataset_path": str(dataset_path),
    "dataset_sha256": file_sha256(dataset_path),
    "candidate_path": str(candidate_path),
    "candidate_checkpoint_sha256": file_sha256(candidate_path),
    "candidate_actor_sha256": final_actor_hash,
    "actor_learning_rate": args.actor_learning_rate,
    "moving_kl_beta": args.moving_kl_beta,
    "max_reference_kl": args.max_reference_kl,
    "epochs": args.epochs,
    "minibatches": args.minibatches,
    "training": training,
    "elapsed_seconds": time.monotonic() - started,
  }
  _atomic_json(output / "training_summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
