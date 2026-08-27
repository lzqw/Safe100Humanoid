"""Learn a filter-free residual adapter from deployable CBF geometry.

The historical 405-D policy cannot observe the toe/riser geometry used by the
runtime filter.  v49 appends five current-state, real-robot-obtainable geometry
coordinates, expands the pretrained actor with exactly zero input columns, and
trains only those columns against successful shielded actions.  Consequently
the candidate remains bit-exact to the base policy whenever the CBF geometry is
inactive, while training still uses the paper's filtered-action distance.
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
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "deployable-cbf-geometry-residual-adapter-v49"
LEGACY_ACTOR_OBSERVATION_DIM = 405
GEOMETRY_OBSERVATION_DIM = 5


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
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--teacher-eta", type=float, default=0.5)
  parser.add_argument("--actor-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--moving-kl-beta", type=float, default=0.1)
  parser.add_argument("--max-reference-kl", type=float, default=0.003)
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v49 training seeds must be comma-separated integers") from exc
  if len(seeds) < 2 or len(set(seeds)) != len(seeds):
    raise ValueError("v49 requires at least two unique training seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v49 expected checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _flat_observations(observations) -> torch.Tensor:
  legacy = observations["actor"]
  geometry = observations["cbf_geometry"]
  if legacy.shape[-1] != LEGACY_ACTOR_OBSERVATION_DIM:
    raise RuntimeError("v49 legacy actor observation is not 405-D")
  if geometry.shape[-1] != GEOMETRY_OBSERVATION_DIM:
    raise RuntimeError("v49 CBF geometry observation is not 5-D")
  return torch.cat((legacy, geometry), dim=-1)


def _actor_observations(flat: torch.Tensor) -> dict[str, torch.Tensor]:
  if flat.shape[-1] != LEGACY_ACTOR_OBSERVATION_DIM + GEOMETRY_OBSERVATION_DIM:
    raise ValueError("v49 flattened actor observation is not 410-D")
  return {
    "actor": flat[:, :LEGACY_ACTOR_OBSERVATION_DIM],
    "cbf_geometry": flat[:, LEGACY_ACTOR_OBSERVATION_DIM:],
  }


def _expand_actor_state(
  source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Copy a 405-D actor into a 410-D actor with zero geometry columns."""
  source_width = int(source["mlp.0.weight"].shape[1])
  target_width = int(target["mlp.0.weight"].shape[1])
  if (source_width, target_width) != (
    LEGACY_ACTOR_OBSERVATION_DIM,
    LEGACY_ACTOR_OBSERVATION_DIM + GEOMETRY_OBSERVATION_DIM,
  ):
    raise RuntimeError(
      f"v49 actor expansion requires 405 -> 410, got {source_width} -> {target_width}"
    )
  if set(source) != set(target):
    raise RuntimeError("v49 actor tensors differ outside their input width")
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
        raise RuntimeError(f"v49 incompatible actor tensor {key!r}")
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
    "source_actor_width": source_width,
    "expanded_actor_width": target_width,
    "new_feature_count": GEOMETRY_OBSERVATION_DIM,
    "legacy_first_layer_copy_max_abs_error": legacy_error,
    "new_first_layer_column_max_abs": zero_error,
    "inactive_geometry_exact_base_policy": True,
    "pi0_exact_preservation_proof": legacy_error == 0.0 and zero_error == 0.0,
  }


def _collect_first_episodes(
  runner,
  base_env,
  action_term,
  *,
  seed: int,
  runtime_filter: bool,
) -> dict[str, Any]:
  action_term.set_runtime_filter_mask(
    torch.full(
      (base_env.num_envs,),
      runtime_filter,
      dtype=torch.bool,
      device=base_env.device,
    )
  )
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
  signature = _initial_state_signature(
    observations,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  actor = runner.alg.actor
  actor.eval()
  active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  success = torch.zeros_like(active)
  fell = torch.zeros_like(active)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  stored_observations: list[torch.Tensor] = []
  stored_nominal_actions: list[torch.Tensor] = []
  stored_safe_actions: list[torch.Tensor] = []
  stored_interventions: list[torch.Tensor] = []
  stored_environment_ids: list[torch.Tensor] = []
  maximum_steps = int(base_env.max_episode_length) + 2
  # ``env.step`` may auto-reset completed worlds and therefore mutates sensor
  # buffers.  ``no_grad`` preserves those ordinary tensors across the paired
  # off/on reset; ``inference_mode`` would permanently tag them as immutable.
  with torch.no_grad():
    for _ in range(maximum_steps):
      flat = _flat_observations(observations)
      actions = actor(observations, stochastic_output=False)
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        stored_observations.append(flat[ids].cpu())
        stored_nominal_actions.append(extras["cbf_nominal_raw_action"][ids].cpu())
        stored_safe_actions.append(extras["cbf_safe_raw_action"][ids].cpu())
        stored_interventions.append(
          extras["cbf_would_intervene"][ids].bool().cpu()
        )
        stored_environment_ids.append(ids.cpu())
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        reached_top = base_env.termination_manager.get_term("reached_top").bool()
        fell_now = extras["online_fell"].bool()
        success[completed] = reached_top[completed]
        fell[completed] = fell_now[completed]
        active &= ~completed
        if not bool(active.any()):
          observations = next_observations
          break
      observations = next_observations
  if bool(active.any()):
    raise RuntimeError("v49 did not finish every first episode")
  dataset = {
    "observations": torch.cat(stored_observations),
    "nominal_actions": torch.cat(stored_nominal_actions),
    "safe_actions": torch.cat(stored_safe_actions),
    "would_intervene": torch.cat(stored_interventions),
    "environment_ids": torch.cat(stored_environment_ids),
  }
  return {
    "seed": seed,
    "runtime_filter": runtime_filter,
    "initial_state_signature": signature,
    "success": success.cpu(),
    "fell": fell.cpu(),
    "steps": steps.cpu(),
    "success_count": int(success.sum()),
    "fall_count": int(fell.sum()),
    "dataset": dataset,
  }


def _distribution_parameters(actor, flat: torch.Tensor):
  actor(_actor_observations(flat), stochastic_output=True)
  return tuple(value for value in actor.output_distribution_params)


def _trust_metrics(
  actor,
  reference_actor,
  observations: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  kl_sum = mean_shift_sum = 0.0
  count = 0
  with torch.inference_mode():
    for start in range(0, len(observations), batch_size):
      batch = observations[start : start + batch_size].to(device)
      reference = _distribution_parameters(reference_actor, batch)
      current = _distribution_parameters(actor, batch)
      kl = diagonal_gaussian_forward_kl(current, reference)
      shift = torch.linalg.vector_norm(current[0] - reference[0], dim=-1)
      kl_sum += float(kl.sum())
      mean_shift_sum += float(shift.sum())
      count += len(batch)
  return {
    "reference_forward_kl": kl_sum / max(1, count),
    "reference_mean_shift": mean_shift_sum / max(1, count),
  }


def _teacher_metrics(
  actor,
  reference_actor,
  dataset: dict[str, torch.Tensor],
  weights: torch.Tensor,
  *,
  eta: float,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  distance_sum = loss_sum = cosine_sum = weight_sum = 0.0
  count = 0
  with torch.inference_mode():
    for start in range(0, len(weights), batch_size):
      stop = min(len(weights), start + batch_size)
      batch = dataset["observations"][start:stop].to(device)
      correction = (
        dataset["safe_actions"][start:stop]
        - dataset["nominal_actions"][start:stop]
      ).to(device)
      reference_mean = _distribution_parameters(reference_actor, batch)[0]
      current_mean = _distribution_parameters(actor, batch)[0]
      target = reference_mean + eta * correction
      effective = weights[start:stop].to(device)
      delta = current_mean - target
      per_loss = F.smooth_l1_loss(
        current_mean, target, reduction="none", beta=0.05
      ).mean(dim=-1)
      shift = current_mean - reference_mean
      cosine = F.cosine_similarity(shift, correction, dim=-1, eps=1.0e-8)
      distance_sum += float(
        (effective * torch.linalg.vector_norm(delta, dim=-1)).sum()
      )
      loss_sum += float((effective * per_loss).sum())
      cosine_sum += float((effective * cosine).sum())
      weight_sum += float(effective.sum())
      count += int((effective > 0.0).sum())
  return {
    "teacher_transition_count": count,
    "teacher_weight_sum": weight_sum,
    "teacher_weighted_distance": distance_sum / max(1.0e-8, weight_sum),
    "teacher_weighted_smooth_l1": loss_sum / max(1.0e-8, weight_sum),
    "teacher_correction_cosine": cosine_sum / max(1.0e-8, weight_sum),
  }


def _train_adapter(
  actor,
  dataset: dict[str, torch.Tensor],
  weights: torch.Tensor,
  trust_observations: torch.Tensor,
  *,
  eta: float,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  epochs: int,
  batch_size: int,
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
  optimizer = torch.optim.Adam([first_layer.weight], lr=learning_rate)
  eligible_indices = (weights > 0.0).nonzero(as_tuple=False).flatten()
  if not len(eligible_indices):
    raise RuntimeError("v49 has no successful shielded intervention targets")
  if not len(trust_observations):
    raise RuntimeError("v49 has no active geometry trust states")
  before_teacher = _teacher_metrics(
    actor,
    reference_actor,
    dataset,
    weights,
    eta=eta,
    device=device,
    batch_size=batch_size,
  )
  before_trust = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=batch_size,
  )
  update_count = 0
  maximum_gradient_norm = 0.0
  for _ in range(epochs):
    permutation = eligible_indices[torch.randperm(len(eligible_indices))]
    for start in range(0, len(permutation), batch_size):
      indices = permutation[start : start + batch_size]
      teacher_obs = dataset["observations"][indices].to(device)
      correction = (
        dataset["safe_actions"][indices]
        - dataset["nominal_actions"][indices]
      ).to(device)
      effective = weights[indices].to(device)
      with torch.no_grad():
        reference_teacher = _distribution_parameters(
          reference_actor, teacher_obs
        )
        target = reference_teacher[0] + eta * correction
      current_teacher = _distribution_parameters(actor, teacher_obs)
      per_teacher_loss = F.smooth_l1_loss(
        current_teacher[0], target, reduction="none", beta=0.05
      ).mean(dim=-1)
      teacher_loss = (effective * per_teacher_loss).sum() / effective.sum().clamp_min(
        1.0e-8
      )
      trust_count = min(batch_size, len(trust_observations))
      trust_indices = torch.randint(len(trust_observations), (trust_count,))
      trust_batch = trust_observations[trust_indices].to(device)
      with torch.no_grad():
        reference_trust = _distribution_parameters(reference_actor, trust_batch)
      current_trust = _distribution_parameters(actor, trust_batch)
      moving_kl = diagonal_gaussian_forward_kl(
        current_trust, reference_trust
      ).mean()
      loss = teacher_loss + moving_kl_beta * moving_kl
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if first_layer.weight.grad is None:
        raise RuntimeError("v49 adapter gradient is missing")
      first_layer.weight.grad[:, :LEGACY_ACTOR_OBSERVATION_DIM] = 0.0
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        [first_layer.weight], max_grad_norm
      )
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v49 adapter gradient is non-finite")
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      optimizer.step()
      with torch.no_grad():
        first_layer.weight[:, :LEGACY_ACTOR_OBSERVATION_DIM].copy_(
          reference_weight[:, :LEGACY_ACTOR_OBSERVATION_DIM]
        )
      update_count += 1

  unprojected_teacher = _teacher_metrics(
    actor,
    reference_actor,
    dataset,
    weights,
    eta=eta,
    device=device,
    batch_size=batch_size,
  )
  unprojected_trust = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=batch_size,
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
  if unprojected_trust["reference_forward_kl"] > max_reference_kl:
    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_scale(middle)
      metrics = _trust_metrics(
        actor,
        reference_actor,
        trust_observations,
        device=device,
        batch_size=batch_size,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    scale = low
    load_scale(scale)
  after_teacher = _teacher_metrics(
    actor,
    reference_actor,
    dataset,
    weights,
    eta=eta,
    device=device,
    batch_size=batch_size,
  )
  after_trust = _trust_metrics(
    actor,
    reference_actor,
    trust_observations,
    device=device,
    batch_size=batch_size,
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
    "trainable_parameter_count": int(first_layer.out_features * 5),
    "optimizer": "adam",
    "optimizer_updates": update_count,
    "epochs": epochs,
    "batch_size": batch_size,
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
    "before": {"teacher": before_teacher, "trust": before_trust},
    "unprojected_after": {
      "teacher": unprojected_teacher,
      "trust": unprojected_trust,
    },
    "after": {"teacher": after_teacher, "trust": after_trust},
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "adapter_interpolation_scale": scale,
      "projection_iterations": projection_iterations,
    },
    "legacy_first_layer_change_max_abs": legacy_error,
    "inactive_geometry_exact_base_policy": legacy_error == 0.0,
  }, optimizer


def _checkpoint_payload(
  source: dict[str, Any],
  state: dict[str, torch.Tensor],
  *,
  metadata: dict[str, Any],
) -> dict[str, Any]:
  output = copy.deepcopy(source)
  output["actor_state_dict"] = {key: value.detach().cpu() for key, value in state.items()}
  output["actor_observation_interface"] = "legacy_405_plus_deployable_cbf_geometry_5"
  infos = dict(output.get("infos") or {})
  infos[METHOD_ID] = metadata
  output["infos"] = infos
  return output


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or args.epochs < 1 or args.batch_size < 1:
    raise ValueError("v49 rollout/training dimensions must be positive")
  if not 0.0 < args.teacher_eta <= 1.0:
    raise ValueError("v49 teacher eta must lie in (0, 1]")
  if not 1.0e-5 <= args.actor_learning_rate <= 0.1:
    raise ValueError("v49 actor learning rate is outside the supported range")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v49 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.02:
    raise ValueError("v49 reference KL cap must lie in (0, 0.02]")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v49 requires a clean committed worktree")
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v49 checkpoint or search configuration is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v49 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v49 base checkpoint SHA-256 differs")
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
      "rollout_conditions": ["filter_off", "filter_on"],
      "actor_observation_interface": "405D proprioception + 5D deployable CBF geometry",
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_deployable_cbf_geometry_observation,
    configure_deployable_cbf_geometry_runner,
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
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v49 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v49 requires the current velocity-CBF action")
  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    initial_state = actor_state(runner.alg.actor)
    initial_actor_hash = actor_state_sha256(initial_state)
    base_expanded_payload = _checkpoint_payload(
      source_payload,
      initial_state,
      metadata={
        "boundary": "zero_adapter_base",
        "geometry": geometry,
        "expansion": expansion,
      },
    )
    base_expanded_path = output / "base_expanded.pt"
    _atomic_torch(base_expanded_path, base_expanded_payload)

    dataset_chunks: dict[str, list[torch.Tensor]] = {
      "observations": [],
      "nominal_actions": [],
      "safe_actions": [],
      "would_intervene": [],
      "environment_ids": [],
    }
    weight_chunks: list[torch.Tensor] = []
    trust_chunks: list[torch.Tensor] = []
    rollout_summaries: list[dict[str, Any]] = []
    global_environment_offset = 0
    rescued_episode_count = 0
    shielded_success_episode_count = 0
    for seed in seeds:
      off = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=False
      )
      on = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=True
      )
      if off["initial_state_signature"] != on["initial_state_signature"]:
        raise RuntimeError("v49 paired filter-on/off initial states differ")
      rescued = on["success"] & ~off["success"]
      shielded_success = on["success"]
      rescued_episode_count += int(rescued.sum())
      shielded_success_episode_count += int(shielded_success.sum())
      on_data = on["dataset"]
      ids = on_data["environment_ids"]
      correction = on_data["safe_actions"] - on_data["nominal_actions"]
      correction_norm = torch.linalg.vector_norm(correction, dim=-1)
      successful_transition = shielded_success[ids]
      rescued_transition = rescued[ids]
      effective = (
        on_data["would_intervene"].float()
        * successful_transition.float()
        * torch.clamp(correction_norm / 0.05, 0.0, 1.0)
        * (1.0 + rescued_transition.float())
      )
      for key in dataset_chunks:
        value = on_data[key]
        if key == "environment_ids":
          value = value + global_environment_offset
        dataset_chunks[key].append(value)
      weight_chunks.append(effective)
      on_active = on_data["observations"][:, -1] > 0.5
      off_data = off["dataset"]
      off_success_transition = off["success"][off_data["environment_ids"]]
      off_active = off_data["observations"][:, -1] > 0.5
      trust_chunks.extend(
        (
          on_data["observations"][on_active],
          off_data["observations"][off_success_transition & off_active],
        )
      )
      rollout_summaries.extend(
        (
          {
            "seed": seed,
            "runtime_filter": False,
            "initial_state_signature": off["initial_state_signature"],
            "success_count": off["success_count"],
            "fall_count": off["fall_count"],
            "transition_count": len(off_data["observations"]),
          },
          {
            "seed": seed,
            "runtime_filter": True,
            "initial_state_signature": on["initial_state_signature"],
            "success_count": on["success_count"],
            "fall_count": on["fall_count"],
            "transition_count": len(on_data["observations"]),
            "rescued_episode_count": int(rescued.sum()),
            "teacher_transition_count": int((effective > 0.0).sum()),
          },
        )
      )
      print(
        json.dumps({"paired_rollout_completed": rollout_summaries[-2:]}),
        flush=True,
      )
      global_environment_offset += args.num_envs

    dataset = {
      key: torch.cat(chunks) for key, chunks in dataset_chunks.items()
    }
    weights = torch.cat(weight_chunks)
    trust_observations = torch.cat(
      [chunk for chunk in trust_chunks if len(chunk)]
    )
    _seed_everything(args.optimization_seed)
    training, optimizer = _train_adapter(
      runner.alg.actor,
      dataset,
      weights,
      trust_observations,
      eta=args.teacher_eta,
      learning_rate=args.actor_learning_rate,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      epochs=args.epochs,
      batch_size=args.batch_size,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    final_state = actor_state(runner.alg.actor)
    candidate_actor_hash = actor_state_sha256(final_state)
    offline_gate_passed = (
      training["after"]["teacher"]["teacher_weighted_distance"]
      < training["before"]["teacher"]["teacher_weighted_distance"]
      and training["after"]["teacher"]["teacher_correction_cosine"] > 0.0
      and training["after"]["trust"]["reference_forward_kl"]
      <= args.max_reference_kl
      and training["legacy_first_layer_change_max_abs"] == 0.0
    )
    candidate_payload = _checkpoint_payload(
      source_payload,
      final_state,
      metadata={
        "boundary": "trained_candidate",
        "geometry": geometry,
        "expansion": expansion,
        "training_seeds": seeds,
        "offline_gate_passed": offline_gate_passed,
      },
    )
    candidate_payload["observable_cbf_adapter_optimizer_state_dict"] = (
      optimizer.state_dict()
    )
    candidate_path = output / "candidate.pt"
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
      "initial_actor_sha256": initial_actor_hash,
      "candidate_actor_sha256": candidate_actor_hash,
      "actor_observation_dim": 410,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "optimization_seed": args.optimization_seed,
      "teacher_eta": args.teacher_eta,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "rescued_episode_count": rescued_episode_count,
      "shielded_success_episode_count": shielded_success_episode_count,
      "training_transition_count": len(dataset["observations"]),
      "active_trust_transition_count": len(trust_observations),
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
