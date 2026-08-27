"""Learn persistent stair geometry from successful filter-free exploration.

v96 combines the low-variance v42 elite self-imitation signal with the v94
persistent next-riser observation.  The pretrained 405-D policy is expanded to
415-D with exact-zero input columns.  Stochastic filter-free first episodes
provide sampled-action targets, but only transitions from successful episodes
are retained.  One full-batch SGD update changes only the ten new geometry
columns, followed by a reference-KL projection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_observable_cbf_adapter_v49 import (
  LEGACY_ACTOR_OBSERVATION_DIM,
  PERSISTENT_GEOMETRY_OBSERVATION_DIM,
  _checkpoint_payload,
  _collect_first_episodes,
  _distribution_parameters,
  _expand_actor_state,
  _trust_metrics,
)
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "persistent-geometry-filter-free-elite-trust-direction-v96"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--reuse-dataset",
    type=Path,
    help="Reuse a completed v96 stochastic rollout dataset without recollecting.",
  )
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=0.1)
  parser.add_argument("--max-reference-kl", type=float, default=5.0e-5)
  parser.add_argument("--metric-batch-size", type=int, default=8192)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v96 training seeds must be comma-separated integers") from exc
  if len(seeds) < 2 or len(set(seeds)) != len(seeds):
    raise ValueError("v96 requires at least two unique training seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v96 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _elite_metrics(
  actor,
  reference_actor,
  observations: torch.Tensor,
  targets: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  distance_sum = loss_sum = reference_distance_sum = 0.0
  dot_sum = shift_square_sum = residual_square_sum = 0.0
  count = 0
  with torch.inference_mode():
    for start in range(0, len(observations), batch_size):
      batch = observations[start : start + batch_size].to(device)
      target = targets[start : start + batch_size].to(device)
      reference_mean = _distribution_parameters(reference_actor, batch)[0]
      current_mean = _distribution_parameters(actor, batch)[0]
      residual = target - reference_mean
      shift = current_mean - reference_mean
      distance_sum += float(torch.linalg.vector_norm(current_mean - target, dim=-1).sum())
      reference_distance_sum += float(
        torch.linalg.vector_norm(reference_mean - target, dim=-1).sum()
      )
      loss_sum += float(
        F.smooth_l1_loss(
          current_mean, target, reduction="none", beta=0.05
        ).mean(dim=-1).sum()
      )
      dot_sum += float((shift * residual).sum())
      shift_square_sum += float(shift.square().sum())
      residual_square_sum += float(residual.square().sum())
      count += len(batch)
  if count < 1:
    raise RuntimeError("v96 has no elite transitions")
  denominator = math.sqrt(shift_square_sum * residual_square_sum)
  return {
    "elite_transition_count": count,
    "elite_target_distance": distance_sum / count,
    "reference_elite_target_distance": reference_distance_sum / count,
    "elite_smooth_l1": loss_sum / count,
    "aggregate_exploration_direction_cosine": (
      dot_sum / denominator if denominator > 1.0e-12 else 0.0
    ),
    "policy_shift_rms": math.sqrt(shift_square_sum / count),
    "exploration_residual_rms": math.sqrt(residual_square_sum / count),
  }


def _train_new_geometry_columns(
  actor,
  observations: torch.Tensor,
  targets: torch.Tensor,
  trust_observations: torch.Tensor,
  *,
  learning_rate: float,
  max_reference_kl: float,
  metric_batch_size: int,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  actor.eval()
  reference_actor = copy.deepcopy(actor).to(device).eval()
  for parameter in reference_actor.parameters():
    parameter.requires_grad_(False)
  for parameter in actor.parameters():
    parameter.requires_grad_(False)
  first_layer = next(
    module for module in actor.mlp if isinstance(module, torch.nn.Linear)
  )
  first_layer.weight.requires_grad_(True)
  reference_weight = first_layer.weight.detach().clone()
  optimizer = torch.optim.SGD([first_layer.weight], lr=learning_rate)
  before = _elite_metrics(
    actor,
    reference_actor,
    observations,
    targets,
    device=device,
    batch_size=metric_batch_size,
  )
  trust_before = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=metric_batch_size,
  )

  batch = observations.to(device)
  target = targets.to(device)
  current_mean = _distribution_parameters(actor, batch)[0]
  loss = F.smooth_l1_loss(current_mean, target, beta=0.05)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  if first_layer.weight.grad is None:
    raise RuntimeError("v96 geometry gradient is missing")
  first_layer.weight.grad[:, :LEGACY_ACTOR_OBSERVATION_DIM] = 0.0
  gradient_norm = torch.nn.utils.clip_grad_norm_([first_layer.weight], max_grad_norm)
  if not bool(torch.isfinite(gradient_norm)):
    raise RuntimeError("v96 geometry gradient is non-finite")
  optimizer.step()
  with torch.no_grad():
    first_layer.weight[:, :LEGACY_ACTOR_OBSERVATION_DIM].copy_(
      reference_weight[:, :LEGACY_ACTOR_OBSERVATION_DIM]
    )
  del batch, target, current_mean, loss

  unprojected = _elite_metrics(
    actor,
    reference_actor,
    observations,
    targets,
    device=device,
    batch_size=metric_batch_size,
  )
  unprojected_trust = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=metric_batch_size,
  )
  proposed_columns = first_layer.weight[
    :, LEGACY_ACTOR_OBSERVATION_DIM:
  ].detach().clone()

  def load_scale(scale: float) -> None:
    with torch.no_grad():
      first_layer.weight.copy_(reference_weight)
      first_layer.weight[:, LEGACY_ACTOR_OBSERVATION_DIM:].copy_(
        proposed_columns * float(scale)
      )

  scale = 1.0
  projection_iterations = 0
  proposal_kl = unprojected_trust["reference_forward_kl"]
  if proposal_kl > max_reference_kl:
    low, high = 0.0, 1.0
  elif proposal_kl > 0.0:
    low, high = 1.0, 2.0
    load_scale(high)
    high_metrics = _trust_metrics(
      actor,
      reference_actor,
      trust_observations,
      device=device,
      batch_size=metric_batch_size,
    )
    while high_metrics["reference_forward_kl"] < max_reference_kl and high < 64.0:
      low = high
      high *= 2.0
      load_scale(high)
      high_metrics = _trust_metrics(
        actor,
        reference_actor,
        trust_observations,
        device=device,
        batch_size=metric_batch_size,
      )
    if high_metrics["reference_forward_kl"] < max_reference_kl:
      scale = high
      load_scale(scale)
      high = low
  else:
    low = high = 0.0
  if high > low:
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_scale(middle)
      metrics = _trust_metrics(
        actor,
        reference_actor,
        trust_observations,
        device=device,
        batch_size=metric_batch_size,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    scale = low
    load_scale(scale)
  after = _elite_metrics(
    actor,
    reference_actor,
    observations,
    targets,
    device=device,
    batch_size=metric_batch_size,
  )
  trust_after = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=metric_batch_size,
  )
  legacy_error = float(
    torch.max(
      torch.abs(
        first_layer.weight[:, :LEGACY_ACTOR_OBSERVATION_DIM]
        - reference_weight[:, :LEGACY_ACTOR_OBSERVATION_DIM]
      )
    )
  )
  return {
    "actor_update_scope": "persistent-geometry-first-layer-columns-only",
    "trainable_parameter_count": (
      first_layer.out_features * PERSISTENT_GEOMETRY_OBSERVATION_DIM
    ),
    "optimizer": "one-full-batch-sgd",
    "optimizer_updates": 1,
    "actor_learning_rate": learning_rate,
    "gradient_norm_pre_clip": float(gradient_norm),
    "before": before,
    "unprojected_after": unprojected,
    "after": after,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "before": trust_before,
      "unprojected_after": unprojected_trust,
      "after": trust_after,
      "adapter_interpolation_scale": scale,
      "direction_extrapolation_allowed": True,
      "maximum_direction_scale": 64.0,
      "projection_iterations": projection_iterations,
    },
    "legacy_first_layer_change_max_abs": legacy_error,
    "inactive_geometry_exact_base_policy": legacy_error == 0.0,
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
  seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or args.metric_batch_size < 1 or args.max_grad_norm <= 0.0:
    raise ValueError("v96 rollout/training dimensions must be positive")
  if not 1.0e-4 <= args.actor_learning_rate <= 1.0:
    raise ValueError("v96 actor learning rate is outside the supported range")
  if not 0.0 < args.max_reference_kl <= 0.001:
    raise ValueError("v96 reference KL cap must lie in (0, 0.001]")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v96 requires a clean committed worktree")
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v96 checkpoint or protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v96 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v96 base checkpoint SHA-256 differs")
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  started = time.monotonic()
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": seeds,
      "rollout_condition": "filter_off_stochastic_first_episode",
      "actor_observation_interface": "405D proprioception + 10D persistent next-riser geometry",
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_deployable_cbf_geometry_runner,
    configure_deployable_cbf_persistent_geometry_observation,
  )
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
  from src.tasks.stairs_cbf.velocity_cbf_action import (
    InstrumentedCurrentVelocityCbfAction,
    configure_v34_cbf,
  )

  _seed_everything(seeds[0])
  env_cfg = load_env_cfg(TASK_ID, play=True)
  shift = configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=False,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  cbf = configure_v34_cbf(
    env_cfg,
    mode=CURRENT_CBF_MODE,
    runtime_filter=False,
    parameters=None,
    measure_compute_time=False,
  )
  reward = configure_paper_dual_reward(
    env_cfg, "raw_moderate", runtime_filter_during_training=True
  )
  geometry = configure_deployable_cbf_persistent_geometry_observation(env_cfg)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = seeds[0]
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v96 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v96 requires the current velocity-CBF action")
  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    initial_state = actor_state(runner.alg.actor)
    initial_actor_hash = actor_state_sha256(initial_state)
    base_expanded_path = output / "base_expanded.pt"
    _atomic_torch(
      base_expanded_path,
      _checkpoint_payload(
        source_payload,
        initial_state,
        method_id=METHOD_ID,
        metadata={
          "boundary": "zero_adapter_base",
          "geometry": geometry,
          "expansion": expansion,
        },
      ),
    )
    distribution = runner.alg.actor.distribution
    exploration_std = (
      distribution.std_param.detach().cpu().tolist()
      if hasattr(distribution, "std_param")
      else None
    )
    reused_dataset: str | None = None
    if args.reuse_dataset is not None:
      dataset_path = args.reuse_dataset.resolve()
      reused = torch.load(dataset_path, map_location="cpu", weights_only=False)
      if reused.get("training_seeds") != seeds:
        raise RuntimeError("v96 reused dataset training seeds differ")
      if reused.get("actor_sha256") != initial_actor_hash:
        raise RuntimeError("v96 reused dataset actor differs")
      all_observations = reused["observations"]
      all_actions = reused["sampled_actions"]
      elite_mask = reused["elite"].bool()
      parent_summary_path = dataset_path.with_name("training_summary.json")
      parent_summary = json.loads(parent_summary_path.read_text())
      rollout_summaries = parent_summary["rollout_summaries"]
      reused_dataset = str(dataset_path)
      parent_outcomes = dataset_path.with_name("rollout_outcomes.csv")
      if parent_outcomes.is_file():
        shutil.copy2(parent_outcomes, output / "rollout_outcomes.csv")
    else:
      observation_chunks: list[torch.Tensor] = []
      action_chunks: list[torch.Tensor] = []
      elite_chunks: list[torch.Tensor] = []
      rollout_summaries = []
      outcome_rows: list[dict[str, Any]] = []
      offset = 0
      for seed in seeds:
        result = _collect_first_episodes(
          runner,
          base_env,
          action_term,
          seed=seed,
          runtime_filter=False,
          stochastic_policy=True,
        )
        data = result["dataset"]
        elite = result["success"][data["environment_ids"]]
        observation_chunks.append(data["observations"])
        action_chunks.append(data["nominal_actions"])
        elite_chunks.append(elite)
        rollout_summaries.append(
          {
            "seed": seed,
            "initial_state_signature": result["initial_state_signature"],
            "success_count": result["success_count"],
            "fall_count": result["fall_count"],
            "transition_count": len(data["observations"]),
            "elite_transition_count": int(elite.sum()),
          }
        )
        for environment_id in range(args.num_envs):
          outcome_rows.append(
            {
              "training_seed": seed,
              "environment_id": environment_id,
              "global_environment_id": offset + environment_id,
              "success": bool(result["success"][environment_id]),
              "fell": bool(result["fell"][environment_id]),
              "steps": int(result["steps"][environment_id]),
            }
          )
        offset += args.num_envs
        print(json.dumps({"rollout_completed": rollout_summaries[-1]}), flush=True)
      all_observations = torch.cat(observation_chunks)
      all_actions = torch.cat(action_chunks)
      elite_mask = torch.cat(elite_chunks)
      dataset_path = output / "elite_dataset.pt"
      _atomic_torch(
        dataset_path,
        {
          "schema_version": 1,
          "method_id": METHOD_ID,
          "training_seeds": seeds,
          "actor_sha256": initial_actor_hash,
          "exploration_std": exploration_std,
          "observations": all_observations,
          "sampled_actions": all_actions,
          "elite": elite_mask,
        },
      )
      _write_outcomes(output / "rollout_outcomes.csv", outcome_rows)
    elite_observations = all_observations[elite_mask]
    elite_actions = all_actions[elite_mask]
    if not len(elite_observations):
      raise RuntimeError("v96 collected no successful filter-free transitions")
    _seed_everything(args.optimization_seed)
    training, optimizer = _train_new_geometry_columns(
      runner.alg.actor,
      elite_observations,
      elite_actions,
      all_observations,
      learning_rate=args.actor_learning_rate,
      max_reference_kl=args.max_reference_kl,
      metric_batch_size=args.metric_batch_size,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    final_state = actor_state(runner.alg.actor)
    candidate_actor_hash = actor_state_sha256(final_state)
    offline_gate_passed = (
      training["after"]["elite_target_distance"]
      < training["before"]["elite_target_distance"]
      and training["after"]["aggregate_exploration_direction_cosine"] > 0.0
      and training["trust_region"]["after"]["reference_forward_kl"]
      <= args.max_reference_kl
      and training["legacy_first_layer_change_max_abs"] == 0.0
    )
    candidate_path = output / "candidate.pt"
    candidate_payload = _checkpoint_payload(
      source_payload,
      final_state,
      method_id=METHOD_ID,
      metadata={
        "boundary": "trained_candidate",
        "geometry": geometry,
        "expansion": expansion,
        "training_seeds": seeds,
        "offline_gate_passed": offline_gate_passed,
      },
    )
    candidate_payload["persistent_elite_optimizer_state_dict"] = optimizer.state_dict()
    _atomic_torch(candidate_path, candidate_payload)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "base_checkpoint_sha256": checkpoint_sha,
      "base_expanded_checkpoint": str(base_expanded_path),
      "base_expanded_checkpoint_sha256": file_sha256(base_expanded_path),
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "elite_dataset": str(dataset_path),
      "elite_dataset_sha256": file_sha256(dataset_path),
      "reused_dataset": reused_dataset,
      "initial_actor_sha256": initial_actor_hash,
      "candidate_actor_sha256": candidate_actor_hash,
      "actor_observation_dim": 415,
      "training_seeds": seeds,
      "optimization_seed": args.optimization_seed,
      "num_envs": args.num_envs,
      "total_initial_episodes": args.num_envs * len(seeds),
      "success_count": sum(item["success_count"] for item in rollout_summaries),
      "fall_count": sum(item["fall_count"] for item in rollout_summaries),
      "training_transition_count": len(all_observations),
      "elite_transition_count": len(elite_observations),
      "exploration_std": exploration_std,
      "rollout_summaries": rollout_summaries,
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "training": training,
      "offline_gate_passed": offline_gate_passed,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
