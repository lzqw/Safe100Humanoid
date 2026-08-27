"""Learn a filter-free residual adapter from deployable CBF geometry.

The historical 405-D policy cannot observe the toe/riser geometry used by the
runtime filter.  v49 appends five current-state, real-robot-obtainable geometry
coordinates, expands the pretrained actor with exactly zero input columns, and
trains only those columns against successful shielded actions.  v93 can instead
append 16 explicit side/phase-conditioned coordinates so opposite filter
corrections do not cancel.  v94 can expose bilateral next-riser geometry
persistently, including before toe-off, so the policy can plan lift earlier.
v109 adds a time-aligned paired trajectory target, while v110 removes its
post-divergence state/action mismatch by querying the CBF on each deployment
trajectory state itself.
v111 additionally signs the full target trace by the paired terminal outcome,
learning toward CBF corrections on rescued pairs and away from them on pairs
where filtering destroys an otherwise successful episode.
Consequently
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
from paired_rescue_v109_math import paired_rescue_action_trace
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
FULL_BATCH_SGD_METHOD_ID = (
  "full-batch-sgd-deployable-cbf-geometry-residual-adapter-v92"
)
CONDITIONAL_FULL_BATCH_SGD_METHOD_ID = (
  "full-batch-sgd-conditional-deployable-cbf-geometry-residual-adapter-v93"
)
PERSISTENT_FULL_BATCH_SGD_METHOD_ID = (
  "full-batch-sgd-persistent-next-riser-geometry-residual-adapter-v94"
)
PAIRED_TRAJECTORY_METHOD_ID = (
  "paired-counterfactual-rescue-trajectory-adapter-v109"
)
DEPLOYMENT_COUNTERFACTUAL_METHOD_ID = (
  "deployment-state-counterfactual-rescue-trajectory-adapter-v110"
)
PAIRED_OUTCOME_CONTRAST_METHOD_ID = (
  "paired-terminal-outcome-contrast-trajectory-adapter-v111"
)
LEGACY_ACTOR_OBSERVATION_DIM = 405
GEOMETRY_OBSERVATION_DIM = 5
PERSISTENT_GEOMETRY_OBSERVATION_DIM = 10
CONDITIONAL_GEOMETRY_OBSERVATION_DIM = 16
SUPPORTED_GEOMETRY_OBSERVATION_DIMS = frozenset(
  (
    GEOMETRY_OBSERVATION_DIM,
    PERSISTENT_GEOMETRY_OBSERVATION_DIM,
    CONDITIONAL_GEOMETRY_OBSERVATION_DIM,
  )
)


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
  parser.add_argument(
    "--geometry-interface",
    choices=("base-5", "persistent-10", "conditional-16"),
    default="base-5",
  )
  parser.add_argument(
    "--teacher-episode-scope",
    choices=("shielded-success", "rescued-only", "discordant-pairs"),
    default="shielded-success",
  )
  parser.add_argument(
    "--teacher-target",
    choices=(
      "instantaneous-filter",
      "paired-trajectory",
      "deployment-counterfactual",
      "paired-outcome-contrast",
    ),
    default="instantaneous-filter",
  )
  parser.add_argument("--paired-pre-horizon", type=int, default=20)
  parser.add_argument("--paired-post-horizon", type=int, default=50)
  parser.add_argument("--paired-pre-decay", type=float, default=0.9)
  parser.add_argument("--actor-learning-rate", type=float, default=1.0e-3)
  parser.add_argument(
    "--adapter-optimizer",
    choices=("adam", "sgd"),
    default="adam",
    help=(
      "Optimize only the appended geometry columns with Adam or "
      "direction-preserving SGD. v93/v94 interfaces require one full SGD batch."
    ),
  )
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
  if geometry.shape[-1] not in SUPPORTED_GEOMETRY_OBSERVATION_DIMS:
    raise RuntimeError("CBF geometry observation is not a supported width")
  return torch.cat((legacy, geometry), dim=-1)


def _actor_observations(flat: torch.Tensor) -> dict[str, torch.Tensor]:
  geometry_dim = flat.shape[-1] - LEGACY_ACTOR_OBSERVATION_DIM
  if geometry_dim not in SUPPORTED_GEOMETRY_OBSERVATION_DIMS:
    raise ValueError("flattened actor observation is not 410-D, 415-D, or 421-D")
  return {
    "actor": flat[:, :LEGACY_ACTOR_OBSERVATION_DIM],
    "cbf_geometry": flat[:, LEGACY_ACTOR_OBSERVATION_DIM:],
  }


def _expand_actor_state(
  source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Copy a 405-D actor into a supported actor with zero geometry columns."""
  source_width = int(source["mlp.0.weight"].shape[1])
  target_width = int(target["mlp.0.weight"].shape[1])
  geometry_dim = target_width - LEGACY_ACTOR_OBSERVATION_DIM
  if (
    source_width != LEGACY_ACTOR_OBSERVATION_DIM
    or geometry_dim not in SUPPORTED_GEOMETRY_OBSERVATION_DIMS
  ):
    raise RuntimeError(
      "adapter actor expansion requires 405 -> 410/415/421, got "
      f"{source_width} -> {target_width}"
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
    "new_feature_count": geometry_dim,
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
  stochastic_policy: bool = False,
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
  stored_episode_steps: list[torch.Tensor] = []
  maximum_steps = int(base_env.max_episode_length) + 2
  # ``env.step`` may auto-reset completed worlds and therefore mutates sensor
  # buffers.  ``no_grad`` preserves those ordinary tensors across the paired
  # off/on reset; ``inference_mode`` would permanently tag them as immutable.
  with torch.no_grad():
    for _ in range(maximum_steps):
      flat = _flat_observations(observations)
      actions = actor(observations, stochastic_output=stochastic_policy)
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
        stored_episode_steps.append(steps[ids].cpu())
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
    "episode_steps": torch.cat(stored_episode_steps),
  }
  return {
    "seed": seed,
    "runtime_filter": runtime_filter,
    "stochastic_policy": stochastic_policy,
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


def _geometry_active(flat: torch.Tensor) -> torch.Tensor:
  """Return the deployable geometric-active mask for either interface."""
  geometry = flat[:, LEGACY_ACTOR_OBSERVATION_DIM:]
  if geometry.shape[-1] == GEOMETRY_OBSERVATION_DIM:
    return geometry[:, 4] > 0.5
  if geometry.shape[-1] == PERSISTENT_GEOMETRY_OBSERVATION_DIM:
    return geometry[:, 4::5].sum(dim=-1) > 0.5
  if geometry.shape[-1] == CONDITIONAL_GEOMETRY_OBSERVATION_DIM:
    return geometry[:, 3::4].sum(dim=-1) > 0.5
  raise ValueError("unknown geometry width for active-state selection")


def _paired_trajectory_rescue_dataset(
  off: dict[str, Any],
  on: dict[str, Any],
  rescued: torch.Tensor,
  *,
  environment_offset: int,
  pre_horizon: int,
  post_horizon: int,
  pre_decay: float,
  target_mode: str = "paired-trajectory",
  episode_signs: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
  """Build episode-balanced rescue targets without mixing state coordinates.

  ``paired-trajectory`` preserves the v109 ablation: filter-on actions are
  indexed onto same-time filter-off observations. ``deployment-counterfactual``
  instead uses the CBF projection computed on each filter-off state itself, so
  state and action remain a valid pair after the two trajectories diverge.
  """
  if target_mode not in (
    "paired-trajectory",
    "deployment-counterfactual",
    "paired-outcome-contrast",
  ):
    raise ValueError("rescue trajectory target mode is unsupported")
  if target_mode == "paired-outcome-contrast":
    if (
      episode_signs is None
      or episode_signs.shape != rescued.shape
      or not bool(((episode_signs == -1) | (episode_signs == 0) | (episode_signs == 1)).all())
      or not torch.equal(episode_signs != 0, rescued.bool())
    ):
      raise ValueError("v111 outcome signs must exactly select +/- discordant pairs")
  elif episode_signs is not None:
    raise ValueError("outcome signs are exclusive to the v111 contrast target")
  off_data = off["dataset"]
  on_data = on["dataset"]
  chunks: dict[str, list[torch.Tensor]] = {
    "observations": [],
    "nominal_actions": [],
    "safe_actions": [],
    "would_intervene": [],
    "environment_ids": [],
  }
  weights: list[torch.Tensor] = []
  episode_summaries: list[dict[str, Any]] = []
  for environment_id in rescued.nonzero(as_tuple=False).flatten().tolist():
    off_rows = (off_data["environment_ids"] == environment_id).nonzero(
      as_tuple=False
    ).flatten()
    on_rows = (on_data["environment_ids"] == environment_id).nonzero(
      as_tuple=False
    ).flatten()
    off_rows = off_rows[off_data["episode_steps"][off_rows].argsort()]
    on_rows = on_rows[on_data["episode_steps"][on_rows].argsort()]
    if not torch.equal(
      off_data["episode_steps"][off_rows], torch.arange(len(off_rows))
    ) or not torch.equal(
      on_data["episode_steps"][on_rows], torch.arange(len(on_rows))
    ):
      raise RuntimeError("v109 paired episode steps are not contiguous")
    if target_mode in (
      "deployment-counterfactual",
      "paired-outcome-contrast",
    ):
      trace = paired_rescue_action_trace(
        off_data["nominal_actions"][off_rows],
        off_data["nominal_actions"][off_rows],
        off_data["safe_actions"][off_rows],
        off_data["would_intervene"][off_rows],
        pre_horizon=pre_horizon,
        post_horizon=post_horizon,
        pre_decay=pre_decay,
      )
      intervention_source = off_data
      intervention_rows = off_rows
    else:
      trace = paired_rescue_action_trace(
        off_data["nominal_actions"][off_rows],
        on_data["nominal_actions"][on_rows],
        on_data["safe_actions"][on_rows],
        on_data["would_intervene"][on_rows],
        pre_horizon=pre_horizon,
        post_horizon=post_horizon,
        pre_decay=pre_decay,
      )
      intervention_source = on_data
      intervention_rows = on_rows
    indices = trace["indices"]
    if not isinstance(indices, torch.Tensor) or not len(indices):
      continue
    selected_off_rows = off_rows[indices]
    correction = trace["corrections"]
    episode_weights = trace["weights"]
    if not isinstance(correction, torch.Tensor) or not isinstance(
      episode_weights, torch.Tensor
    ):
      raise RuntimeError("v109 trace tensors are missing")
    outcome_sign = 1
    if target_mode == "paired-outcome-contrast":
      assert episode_signs is not None
      outcome_sign = int(episode_signs[environment_id])
      correction = correction * float(outcome_sign)
    nominal = off_data["nominal_actions"][selected_off_rows]
    chunks["observations"].append(off_data["observations"][selected_off_rows])
    chunks["nominal_actions"].append(nominal)
    chunks["safe_actions"].append(nominal + correction)
    chunks["would_intervene"].append(
      intervention_source["would_intervene"][intervention_rows[indices]]
    )
    chunks["environment_ids"].append(
      torch.full(
        (len(indices),),
        int(environment_offset + environment_id),
        dtype=torch.long,
      )
    )
    weights.append(episode_weights)
    norms = torch.linalg.vector_norm(correction, dim=-1)
    episode_summaries.append(
      {
        "environment_id": int(environment_id),
        "first_intervention_step": int(trace["first_intervention_step"]),
        "shared_length": int(trace["shared_length"]),
        "teacher_transition_count": len(indices),
        "pre_transition_count": int(trace["pre_transition_count"]),
        "post_transition_count": int(trace["post_transition_count"]),
        "correction_norm_mean": float(norms.mean()),
        "correction_norm_max": float(norms.max()),
        "episode_weight_sum": float(episode_weights.sum()),
        "target_mode": target_mode,
        "paired_outcome_sign": outcome_sign,
      }
    )
  if not weights:
    empty = off_data["observations"].new_empty(
      (0, off_data["observations"].shape[1])
    )
    empty_actions = off_data["nominal_actions"].new_empty(
      (0, off_data["nominal_actions"].shape[1])
    )
    dataset = {
      "observations": empty,
      "nominal_actions": empty_actions,
      "safe_actions": empty_actions.clone(),
      "would_intervene": torch.empty(0, dtype=torch.bool),
      "environment_ids": torch.empty(0, dtype=torch.long),
    }
    return dataset, torch.empty(0), {
      "target_mode": target_mode,
      "positive_episode_count": 0,
      "negative_episode_count": 0,
      "episodes": [],
    }
  dataset = {key: torch.cat(value) for key, value in chunks.items()}
  combined_weights = torch.cat(weights)
  return dataset, combined_weights, {
    "target_mode": target_mode,
    "episode_count": len(episode_summaries),
    "positive_episode_count": sum(
      row["paired_outcome_sign"] > 0 for row in episode_summaries
    ),
    "negative_episode_count": sum(
      row["paired_outcome_sign"] < 0 for row in episode_summaries
    ),
    "teacher_transition_count": len(combined_weights),
    "pre_transition_count": sum(
      row["pre_transition_count"] for row in episode_summaries
    ),
    "post_transition_count": sum(
      row["post_transition_count"] for row in episode_summaries
    ),
    "episode_weight_sum_min": min(
      row["episode_weight_sum"] for row in episode_summaries
    ),
    "episode_weight_sum_max": max(
      row["episode_weight_sum"] for row in episode_summaries
    ),
    "episodes": episode_summaries,
  }


def _build_adapter_optimizer(
  parameters: list[torch.nn.Parameter],
  *,
  optimizer_name: str,
  learning_rate: float,
) -> torch.optim.Optimizer:
  if optimizer_name == "adam":
    return torch.optim.Adam(parameters, lr=learning_rate)
  if optimizer_name == "sgd":
    return torch.optim.SGD(parameters, lr=learning_rate)
  raise ValueError(f"unknown v49-v93 adapter optimizer {optimizer_name!r}")


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
  optimizer_name: str,
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
  optimizer = _build_adapter_optimizer(
    [first_layer.weight],
    optimizer_name=optimizer_name,
    learning_rate=learning_rate,
  )
  eligible_indices = (weights > 0.0).nonzero(as_tuple=False).flatten()
  if not len(eligible_indices):
    raise RuntimeError("v49 has no successful shielded intervention targets")
  if optimizer_name == "sgd" and (
    epochs != 1 or batch_size < len(eligible_indices)
  ):
    raise ValueError(
      "v92/v93 SGD requires exactly one epoch and one full eligible batch"
    )
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
    "actor_update_scope": "new-geometry-first-layer-input-columns-only",
    "trainable_parameter_count": int(
      first_layer.out_features
      * (first_layer.in_features - LEGACY_ACTOR_OBSERVATION_DIM)
    ),
    "optimizer": optimizer_name,
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
  method_id: str = METHOD_ID,
  metadata: dict[str, Any],
) -> dict[str, Any]:
  output = copy.deepcopy(source)
  output["actor_state_dict"] = {key: value.detach().cpu() for key, value in state.items()}
  geometry_dim = int(state["mlp.0.weight"].shape[1]) - LEGACY_ACTOR_OBSERVATION_DIM
  output["actor_observation_interface"] = (
    f"legacy_405_plus_deployable_cbf_geometry_{geometry_dim}"
  )
  infos = dict(output.get("infos") or {})
  infos[method_id] = metadata
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
  if args.geometry_interface != "base-5" and args.adapter_optimizer != "sgd":
    raise ValueError("v93/v94 geometry requires direction-preserving SGD")
  trajectory_teacher = args.teacher_target in (
    "paired-trajectory",
    "deployment-counterfactual",
    "paired-outcome-contrast",
  )
  contrast_teacher = args.teacher_target == "paired-outcome-contrast"
  if trajectory_teacher and (
    args.geometry_interface != "persistent-10"
    or args.teacher_episode_scope
    != ("discordant-pairs" if contrast_teacher else "rescued-only")
    or args.adapter_optimizer != "sgd"
    or args.paired_pre_horizon < 0
    or args.paired_post_horizon < 0
    or not 0.0 < args.paired_pre_decay <= 1.0
  ):
    raise ValueError(
      "v109-v111 trajectory targets require their paired scope, persistent SGD, and valid trace"
    )
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
  if args.teacher_target == "paired-outcome-contrast":
    method_id = PAIRED_OUTCOME_CONTRAST_METHOD_ID
  elif args.teacher_target == "deployment-counterfactual":
    method_id = DEPLOYMENT_COUNTERFACTUAL_METHOD_ID
  elif args.teacher_target == "paired-trajectory":
    method_id = PAIRED_TRAJECTORY_METHOD_ID
  elif args.geometry_interface == "persistent-10":
    method_id = PERSISTENT_FULL_BATCH_SGD_METHOD_ID
  elif args.geometry_interface == "conditional-16":
    method_id = CONDITIONAL_FULL_BATCH_SGD_METHOD_ID
  elif args.adapter_optimizer == "sgd":
    method_id = FULL_BATCH_SGD_METHOD_ID
  else:
    method_id = METHOD_ID
  geometry_dim = {
    "base-5": GEOMETRY_OBSERVATION_DIM,
    "persistent-10": PERSISTENT_GEOMETRY_OBSERVATION_DIM,
    "conditional-16": CONDITIONAL_GEOMETRY_OBSERVATION_DIM,
  }[args.geometry_interface]
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": seeds,
      "rollout_conditions": ["filter_off", "filter_on"],
      "teacher_episode_scope": args.teacher_episode_scope,
      "teacher_target": args.teacher_target,
      "actor_observation_interface": (
        f"405D proprioception + {geometry_dim}D deployable CBF geometry"
      ),
    },
  )

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_deployable_cbf_conditional_geometry_observation,
    configure_deployable_cbf_geometry_observation,
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
  if args.geometry_interface == "persistent-10":
    geometry = configure_deployable_cbf_persistent_geometry_observation(env_cfg)
  elif args.geometry_interface == "conditional-16":
    geometry = configure_deployable_cbf_conditional_geometry_observation(env_cfg)
  else:
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
      method_id=method_id,
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
    harmed_episode_count = 0
    shielded_success_episode_count = 0
    paired_trace_summaries: list[dict[str, Any]] = []
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
      harmed = off["success"] & ~on["success"]
      shielded_success = on["success"]
      rescued_episode_count += int(rescued.sum())
      harmed_episode_count += int(harmed.sum())
      shielded_success_episode_count += int(shielded_success.sum())
      on_data = on["dataset"]
      ids = on_data["environment_ids"]
      rescued_transition = rescued[ids]
      if trajectory_teacher:
        selected_episodes = rescued
        episode_signs = None
        if contrast_teacher:
          selected_episodes = rescued | harmed
          episode_signs = rescued.to(torch.int8) - harmed.to(torch.int8)
        paired_dataset, effective, paired_summary = (
          _paired_trajectory_rescue_dataset(
            off,
            on,
            selected_episodes,
            environment_offset=global_environment_offset,
            pre_horizon=args.paired_pre_horizon,
            post_horizon=args.paired_post_horizon,
            pre_decay=args.paired_pre_decay,
            target_mode=args.teacher_target,
            episode_signs=episode_signs,
          )
        )
        if len(effective):
          for key in dataset_chunks:
            dataset_chunks[key].append(paired_dataset[key])
          weight_chunks.append(effective)
        paired_trace_summaries.append(
          {"seed": seed, **paired_summary}
        )
      else:
        correction = on_data["safe_actions"] - on_data["nominal_actions"]
        correction_norm = torch.linalg.vector_norm(correction, dim=-1)
        successful_transition = shielded_success[ids]
        teacher_episode_transition = (
          rescued_transition
          if args.teacher_episode_scope == "rescued-only"
          else successful_transition
        )
        effective = (
          on_data["would_intervene"].float()
          * teacher_episode_transition.float()
          * torch.clamp(correction_norm / 0.05, 0.0, 1.0)
          * (1.0 + rescued_transition.float())
        )
        for key in dataset_chunks:
          value = on_data[key]
          if key == "environment_ids":
            value = value + global_environment_offset
          dataset_chunks[key].append(value)
        weight_chunks.append(effective)
      on_active = _geometry_active(on_data["observations"])
      off_data = off["dataset"]
      off_success_transition = off["success"][off_data["environment_ids"]]
      off_active = _geometry_active(off_data["observations"])
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
            "harmed_episode_count": int(harmed.sum()),
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
      optimizer_name=args.adapter_optimizer,
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
      method_id=method_id,
      metadata={
        "boundary": "trained_candidate",
        "geometry": geometry,
        "expansion": expansion,
        "training_seeds": seeds,
        "teacher_target": args.teacher_target,
        "paired_trace": {
          "pre_horizon": args.paired_pre_horizon,
          "post_horizon": args.paired_post_horizon,
          "pre_decay": args.paired_pre_decay,
        }
        if trajectory_teacher
        else None,
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
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "base_checkpoint_sha256": checkpoint_sha,
      "base_expanded_checkpoint": str(base_expanded_path),
      "base_expanded_checkpoint_sha256": file_sha256(base_expanded_path),
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "initial_actor_sha256": initial_actor_hash,
      "candidate_actor_sha256": candidate_actor_hash,
      "actor_observation_dim": LEGACY_ACTOR_OBSERVATION_DIM + geometry_dim,
      "geometry_interface": args.geometry_interface,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "optimization_seed": args.optimization_seed,
      "teacher_eta": args.teacher_eta,
      "teacher_episode_scope": args.teacher_episode_scope,
      "teacher_target": args.teacher_target,
      "paired_trace": {
        "pre_horizon": args.paired_pre_horizon,
        "post_horizon": args.paired_post_horizon,
        "pre_decay": args.paired_pre_decay,
        "seed_summaries": paired_trace_summaries,
      }
      if trajectory_teacher
      else None,
      "actor_learning_rate": args.actor_learning_rate,
      "adapter_optimizer": args.adapter_optimizer,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl": args.max_reference_kl,
      "rescued_episode_count": rescued_episode_count,
      "harmed_episode_count": harmed_episode_count,
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
