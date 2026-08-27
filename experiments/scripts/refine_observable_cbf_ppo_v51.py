"""Paper-style paired CBF-dual PPO for the deployable geometry adapter.

v49/v50 showed that the missing CBF geometry is observable and useful, but
direct safe-action regression did not survive an untouched gate.  v51 keeps
the exact 405-D base policy and trains only the five appended input columns
using paired filter-on/off GAE returns from the paper-aligned bounded reward.
"""

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
from refine_multi_rollout_gae_v46 import _collect_rollout, _policy_metrics
from refine_observable_cbf_adapter_v49 import (
  GEOMETRY_OBSERVATION_DIM,
  LEGACY_ACTOR_OBSERVATION_DIM,
  _expand_actor_state,
)
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "observable-cbf-geometry-paired-dual-gae-v51"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--rollout-steps", type=int, default=512)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=0.01)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=0.001)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument("--max-grad-norm", type=float, default=10.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v51 training seeds must be comma-separated integers") from exc
  if len(seeds) < 3 or len(set(seeds)) != len(seeds):
    raise ValueError("v51 requires at least three unique training seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v51 expected checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _expand_critic_state(
  source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Append five zero columns to the online critic's legacy prefix."""
  source_width = int(source["mlp.0.weight"].shape[1])
  target_width = int(target["mlp.0.weight"].shape[1])
  if target_width - source_width != GEOMETRY_OBSERVATION_DIM:
    raise RuntimeError(
      f"v51 critic expansion requires five columns, got {source_width} -> {target_width}"
    )
  if set(source) != set(target):
    raise RuntimeError("v51 critic tensors differ outside their input width")
  expanded = {key: value.detach().clone() for key, value in target.items()}
  with torch.no_grad():
    for key, target_value in expanded.items():
      source_value = source[key].to(target_value.device, target_value.dtype)
      if key == "mlp.0.weight":
        target_value.zero_()
        target_value[:, :source_width].copy_(source_value)
      elif key.startswith("obs_normalizer._") and target_value.ndim == 2:
        if key.endswith("_var") or key.endswith("_std"):
          target_value.fill_(1.0)
        else:
          target_value.zero_()
        target_value[:, :source_width].copy_(source_value)
      elif source_value.shape == target_value.shape:
        target_value.copy_(source_value)
      else:
        raise RuntimeError(f"v51 incompatible critic tensor {key!r}")
  legacy_error = float(
    torch.max(
      torch.abs(
        expanded["mlp.0.weight"][:, :source_width]
        - source["mlp.0.weight"].to(expanded["mlp.0.weight"].device)
      )
    )
  )
  zero_error = float(expanded["mlp.0.weight"][:, source_width:].abs().max())
  return expanded, {
    "source_critic_width": source_width,
    "expanded_critic_width": target_width,
    "legacy_first_layer_copy_max_abs_error": legacy_error,
    "new_first_layer_column_max_abs": zero_error,
    "exact_prefix_expansion": legacy_error == 0.0 and zero_error == 0.0,
  }


def _adapter_gradient_diagnostics(
  gradients: list[torch.Tensor], labels: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[torch.Tensor]]:
  flattened = torch.stack([gradient.flatten() for gradient in gradients])
  norms = torch.linalg.vector_norm(flattened, dim=1)
  normalized = flattened / norms.clamp_min(1.0e-12).unsqueeze(1)
  cosines = normalized @ normalized.T
  off_diagonal = cosines[
    ~torch.eye(len(gradients), dtype=torch.bool, device=cosines.device)
  ]
  paired: list[torch.Tensor] = []
  paired_labels: list[int] = []
  paired_on_off_cosines: list[float] = []
  for seed in sorted({int(label["seed"]) for label in labels}):
    indices = [
      index for index, label in enumerate(labels) if int(label["seed"]) == seed
    ]
    conditions = {bool(labels[index]["runtime_filter"]) for index in indices}
    if len(indices) != 2 or conditions != {False, True}:
      raise RuntimeError("v51 requires one filter-on/off rollout per seed")
    paired.append(0.5 * (gradients[indices[0]] + gradients[indices[1]]))
    paired_labels.append(seed)
    paired_on_off_cosines.append(float(cosines[indices[0], indices[1]]))
  paired_flattened = torch.stack([gradient.flatten() for gradient in paired])
  paired_norms = torch.linalg.vector_norm(paired_flattened, dim=1)
  paired_normalized = paired_flattened / paired_norms.clamp_min(1.0e-12).unsqueeze(1)
  paired_cosines = paired_normalized @ paired_normalized.T
  paired_off_diagonal = paired_cosines[
    ~torch.eye(len(paired), dtype=torch.bool, device=paired_cosines.device)
  ]
  return {
    "per_batch_adapter_gradient_norm": norms.tolist(),
    "pairwise_adapter_gradient_cosine_min": float(off_diagonal.min()),
    "pairwise_adapter_gradient_cosine_mean": float(off_diagonal.mean()),
    "paired_filter_on_off_adapter_gradient_cosines": paired_on_off_cosines,
    "paired_filter_on_off_adapter_gradient_cosine_mean": sum(
      paired_on_off_cosines
    )
    / len(paired_on_off_cosines),
    "paired_seed_labels": paired_labels,
    "paired_seed_adapter_gradient_norm": paired_norms.tolist(),
    "paired_seed_adapter_gradient_cosine_min": float(
      paired_off_diagonal.min()
    ),
    "paired_seed_adapter_gradient_cosine_mean": float(
      paired_off_diagonal.mean()
    ),
  }, paired


def _robust_adapter_ppo_step(
  actor,
  batches: list[Any],
  labels: list[dict[str, Any]],
  *,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  clip_ratio: float,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

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
  before = _policy_metrics(
    actor, reference_actor, batches, labels=labels, clip_ratio=clip_ratio
  )
  adapter_gradients: list[torch.Tensor] = []
  policy_losses: list[float] = []
  for batch in batches:
    observations = batch.observations.flatten(0, 1)
    actions = batch.actions.flatten(0, 1)
    old_log_prob = batch.actions_log_prob.flatten(0, 1).squeeze(-1)
    advantages = batch.advantages.flatten().detach()
    with torch.no_grad():
      reference_actor(observations, stochastic_output=True)
      reference_parameters = tuple(
        value.detach() for value in reference_actor.output_distribution_params
      )
    actor(observations, stochastic_output=True)
    current_parameters = tuple(actor.output_distribution_params)
    ratio = torch.exp(actor.get_output_log_prob(actions) - old_log_prob)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.minimum(
      advantages * ratio, advantages * clipped
    ).mean()
    moving_kl = diagonal_gaussian_forward_kl(
      current_parameters, reference_parameters
    ).mean()
    gradient = torch.autograd.grad(
      policy_loss + moving_kl_beta * moving_kl, first_layer.weight
    )[0]
    adapter_gradient = gradient[:, LEGACY_ACTOR_OBSERVATION_DIM:].detach()
    if not bool(torch.isfinite(adapter_gradient).all()):
      raise RuntimeError("v51 adapter gradient is non-finite")
    adapter_gradients.append(adapter_gradient)
    policy_losses.append(float(policy_loss.detach()))
  gradient_metrics, paired_gradients = _adapter_gradient_diagnostics(
    adapter_gradients, labels
  )
  aggregate = torch.stack(paired_gradients).median(dim=0).values
  first_layer.weight.grad = torch.zeros_like(first_layer.weight)
  first_layer.weight.grad[:, LEGACY_ACTOR_OBSERVATION_DIM:] = aggregate
  aggregate_norm = torch.nn.utils.clip_grad_norm_([first_layer.weight], max_grad_norm)
  if not bool(torch.isfinite(aggregate_norm)):
    raise RuntimeError("v51 aggregate adapter gradient is non-finite")
  optimizer.step()
  with torch.no_grad():
    first_layer.weight[:, :LEGACY_ACTOR_OBSERVATION_DIM].copy_(
      reference_weight[:, :LEGACY_ACTOR_OBSERVATION_DIM]
    )
  unprojected = _policy_metrics(
    actor, reference_actor, batches, labels=labels, clip_ratio=clip_ratio
  )
  proposal_columns = first_layer.weight[
    :, LEGACY_ACTOR_OBSERVATION_DIM:
  ].detach().clone()
  reference_columns = reference_weight[
    :, LEGACY_ACTOR_OBSERVATION_DIM:
  ].detach().clone()

  def load_scale(scale: float) -> None:
    with torch.no_grad():
      first_layer.weight.copy_(reference_weight)
      first_layer.weight[:, LEGACY_ACTOR_OBSERVATION_DIM:].copy_(
        reference_columns + float(scale) * (proposal_columns - reference_columns)
      )

  scale = 1.0
  projection_iterations = 0
  if unprojected["reference_forward_kl"] > max_reference_kl:
    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_scale(middle)
      metrics = _policy_metrics(
        actor, reference_actor, batches, labels=labels, clip_ratio=clip_ratio
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    scale = low
    load_scale(scale)
  after = _policy_metrics(
    actor, reference_actor, batches, labels=labels, clip_ratio=clip_ratio
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
    "actor_update_scope": "five-new-first-layer-input-columns-only",
    "trainable_parameter_count": first_layer.out_features
    * GEOMETRY_OBSERVATION_DIM,
    "optimizer": "sgd",
    "optimizer_updates": 1,
    "gradient_aggregation": "paired-on-off-mean_then-coordinate-median",
    "per_batch_policy_loss_before": policy_losses,
    **gradient_metrics,
    "aggregate_gradient_norm_pre_clip": float(aggregate_norm),
    "before": before,
    "unprojected_after": unprojected,
    "after": after,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "adapter_interpolation_scale": scale,
      "projection_iterations": projection_iterations,
    },
    "legacy_first_layer_change_max_abs": legacy_error,
    "inactive_geometry_exact_base_policy": legacy_error == 0.0,
  }, optimizer


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or args.rollout_steps < 64:
    raise ValueError("v51 rollout dimensions are too small")
  if not 1.0e-5 <= args.actor_learning_rate <= 0.1:
    raise ValueError("v51 actor learning rate is outside the supported range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v51 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.02:
    raise ValueError("v51 reference KL cap must lie in (0, 0.02]")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v51 requires a clean committed worktree")
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v51 checkpoint or search configuration is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v51 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v51 base checkpoint SHA-256 differs")
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
      "rollout_conditions": ["filter_on", "filter_off"],
      "optimization_seed": args.optimization_seed,
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_deployable_cbf_geometry_observation,
    configure_deployable_cbf_geometry_runner,
  )
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
  from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner
  from src.tasks.stairs_cbf.velocity_cbf_action import configure_v34_cbf

  _seed_everything(seeds[0])
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
  cbf = configure_v34_cbf(
    env_cfg,
    mode=CURRENT_CBF_MODE,
    runtime_filter=True,
    parameters=None,
    measure_compute_time=False,
  )
  reward = configure_paper_dual_reward(
    env_cfg, "raw_moderate", runtime_filter_during_training=True
  )
  geometry = configure_deployable_cbf_geometry_observation(env_cfg)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = seeds[0]
  agent_cfg.num_steps_per_env = args.rollout_steps
  _configure_algorithm(agent_cfg, "A0", preflight=False)
  agent_cfg.algorithm.minimum_std = 0.05
  agent_cfg.algorithm.maximum_std = 0.05
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner = CbfTeacherV30Runner(
    env, asdict(agent_cfg), log_dir=None, device=args.device
  )
  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_actor, actor_expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    expanded_critic, critic_expansion = _expand_critic_state(
      source_payload["critic_state_dict"], runner.alg.critic.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_actor, strict=True)
    runner.alg.critic.load_state_dict(expanded_critic, strict=True)
    runner.alg._std_initialized = False
    runner.alg.initialize_online_std()
    runner.alg.freeze_round_reference()
    initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
    action_term = base_env.action_manager.get_term("joint_pos")
    batches: list[Any] = []
    rollout_summaries: list[dict[str, Any]] = []
    signatures: dict[int, str] = {}
    for seed in seeds:
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
          raise RuntimeError("v51 paired initial states differ")
        batches.append(batch)
        rollout_summaries.append(rollout)
        print(json.dumps({"rollout_completed": rollout}), flush=True)
    _seed_everything(args.optimization_seed)
    training, optimizer = _robust_adapter_ppo_step(
      runner.alg.actor,
      batches,
      rollout_summaries,
      learning_rate=args.actor_learning_rate,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      clip_ratio=args.clip_ratio,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    after = training["after"]
    offline_gate_passed = (
      after["positive_batch_count"] == after["batch_count"]
      and after["mean_filter_on_surrogate_gain"] > 0.0
      and after["mean_filter_off_surrogate_gain"] > 0.0
      and after["reference_forward_kl"] <= args.max_reference_kl
      and training["legacy_first_layer_change_max_abs"] == 0.0
    )
    final_state = actor_state(runner.alg.actor)
    final_actor_hash = actor_state_sha256(final_state)
    candidate_payload = copy.deepcopy(source_payload)
    candidate_payload["actor_state_dict"] = {
      key: value.detach().cpu() for key, value in final_state.items()
    }
    candidate_payload["actor_observation_interface"] = (
      "legacy_405_plus_deployable_cbf_geometry_5"
    )
    candidate_payload["observable_cbf_ppo_optimizer_state_dict"] = (
      optimizer.state_dict()
    )
    infos = dict(candidate_payload.get("infos") or {})
    infos[METHOD_ID] = {
      "source_git_commit": _git(repo, "rev-parse", "HEAD"),
      "training_seeds": seeds,
      "optimization_seed": args.optimization_seed,
      "offline_gate_passed": offline_gate_passed,
    }
    candidate_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, candidate_payload)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "rollout_steps": args.rollout_steps,
      "rollout_batch_count": len(batches),
      "training_transition_count": len(batches)
      * args.num_envs
      * args.rollout_steps,
      "rollout_summaries": rollout_summaries,
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": actor_expansion,
      "critic_expansion": critic_expansion,
      "base_checkpoint_sha256": checkpoint_sha,
      "initial_actor_sha256": initial_actor_hash,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "candidate_actor_sha256": final_actor_hash,
      "optimization_seed": args.optimization_seed,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "clip_ratio": args.clip_ratio,
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
