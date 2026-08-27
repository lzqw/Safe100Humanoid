"""Learn a paired treatment-effect gate and a deployable CBF residual.

v111/v112 showed that globally adding rescued CBF corrections and subtracting
harmed corrections nearly cancels.  v113 separates the two questions:

1. a treatment gate predicts from the filter-off state whether enabling the
   CBF changes the paired terminal outcome from failure to success; and
2. a bounded residual predicts the same-state CBF correction only on those
   matched-rescue trajectories.

The v79 task actor remains frozen.  At deployment the analytic runtime filter
is disabled and the learned correction is applied only when the gate is both
confident and deployable CBF geometry is active.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from refine_observable_cbf_adapter_v49 import (
  LEGACY_ACTOR_OBSERVATION_DIM,
  PERSISTENT_GEOMETRY_OBSERVATION_DIM,
  _actor_observations,
  _collect_first_episodes,
  _expand_actor_state,
  _paired_trajectory_rescue_dataset,
)
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from train_learned_residual_v97 import _base_hidden
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "paired-treatment-gated-cbf-residual-v113"
BASE_HIDDEN_DIM = 128
ACTION_DIM = 12
FEATURE_DIM = BASE_HIDDEN_DIM + PERSISTENT_GEOMETRY_OBSERVATION_DIM + ACTION_DIM


class PairedTreatmentGatedResidual(torch.nn.Module):
  """Independent treatment gate and bounded action-residual heads."""

  def __init__(
    self,
    max_residual: float,
    gate_hidden_dims: tuple[int, int] = (64, 32),
    residual_hidden_dims: tuple[int, int] = (128, 64),
  ) -> None:
    super().__init__()
    self.max_residual = float(max_residual)
    self.gate = torch.nn.Sequential(
      torch.nn.Linear(FEATURE_DIM, gate_hidden_dims[0]),
      torch.nn.ELU(),
      torch.nn.Linear(gate_hidden_dims[0], gate_hidden_dims[1]),
      torch.nn.ELU(),
      torch.nn.Linear(gate_hidden_dims[1], 1),
    )
    self.residual = torch.nn.Sequential(
      torch.nn.Linear(FEATURE_DIM, residual_hidden_dims[0]),
      torch.nn.ELU(),
      torch.nn.Linear(residual_hidden_dims[0], residual_hidden_dims[1]),
      torch.nn.ELU(),
      torch.nn.Linear(residual_hidden_dims[1], ACTION_DIM),
    )
    final = self.residual[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)

  def gate_probability(self, features: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.gate(features).squeeze(-1))

  def residual_action(self, features: torch.Tensor) -> torch.Tensor:
    raw = self.residual(features)
    return self.max_residual * torch.tanh(raw / self.max_residual)


def deployable_gate(
  probability: torch.Tensor,
  geometry_active: torch.Tensor,
  *,
  threshold: float,
) -> torch.Tensor:
  """Return a continuous high-confidence gate with an exact zero region."""
  if probability.ndim != 1 or geometry_active.shape != probability.shape:
    raise ValueError("v113 gate inputs must be aligned vectors")
  if geometry_active.dtype != torch.bool or not 0.0 < threshold < 1.0:
    raise ValueError("v113 gate mask or threshold is invalid")
  above = torch.clamp(
    (probability - float(threshold)) / (1.0 - float(threshold)),
    0.0,
    1.0,
  )
  return above * geometry_active.to(probability.dtype)


def balanced_episode_weights(
  labels: torch.Tensor,
  environment_ids: torch.Tensor,
) -> torch.Tensor:
  """Give both classes half the mass and every episode equal class mass."""
  if (
    labels.ndim != 1
    or environment_ids.shape != labels.shape
    or labels.dtype != torch.bool
    or environment_ids.dtype != torch.long
    or not len(labels)
  ):
    raise ValueError("v113 gate labels and episode ids are invalid")
  weights = torch.zeros(len(labels), dtype=torch.float32)
  episode_rows: list[tuple[torch.Tensor, bool]] = []
  for environment_id in environment_ids.unique(sorted=True).tolist():
    rows = (environment_ids == int(environment_id)).nonzero(
      as_tuple=False
    ).flatten()
    episode_labels = labels[rows]
    if not bool((episode_labels == episode_labels[0]).all()):
      raise ValueError("v113 treatment label must be constant within an episode")
    episode_rows.append((rows, bool(episode_labels[0])))
  positive_count = sum(label for _, label in episode_rows)
  negative_count = len(episode_rows) - positive_count
  if positive_count == 0 or negative_count == 0:
    raise ValueError("v113 gate requires positive and negative paired outcomes")
  for rows, positive in episode_rows:
    class_episode_count = positive_count if positive else negative_count
    weights[rows] = 0.5 / float(class_episode_count * len(rows))
  return weights


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--screen-seed", type=int, required=True)
  parser.add_argument("--screen-envs", type=int, default=64)
  parser.add_argument("--paired-pre-horizon", type=int, default=20)
  parser.add_argument("--paired-post-horizon", type=int, default=50)
  parser.add_argument("--paired-pre-decay", type=float, default=0.9)
  parser.add_argument("--teacher-eta", type=float, default=0.25)
  parser.add_argument("--max-residual", type=float, default=0.25)
  parser.add_argument("--gate-threshold", type=float, default=0.5)
  parser.add_argument("--gate-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--gate-epochs", type=int, default=20)
  parser.add_argument("--residual-epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--residual-trust-weight", type=float, default=0.25)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v113 seeds must be comma-separated integers") from exc
  if len(seeds) < 3 or len(set(seeds)) != len(seeds):
    raise ValueError("v113 requires at least three unique paired seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v113 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _geometry_active_from_observations(observations) -> torch.Tensor:
  geometry = observations["cbf_geometry"]
  if geometry.shape[-1] != PERSISTENT_GEOMETRY_OBSERVATION_DIM:
    raise RuntimeError("v113 requires 10-D persistent CBF geometry")
  return geometry[:, 4::5].sum(dim=-1) > 0.5


def _features_from_observations(actor, observations) -> torch.Tensor:
  action = actor(observations, stochastic_output=False)
  hidden = _base_hidden(actor, observations)
  geometry = observations["cbf_geometry"]
  features = torch.cat((hidden, geometry, action), dim=-1)
  if features.shape[-1] != FEATURE_DIM:
    raise RuntimeError("v113 deployable feature width differs")
  return features


def _features_from_flat(
  actor,
  flat: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> torch.Tensor:
  chunks: list[torch.Tensor] = []
  actor.eval()
  with torch.inference_mode():
    for start in range(0, len(flat), batch_size):
      observations = _actor_observations(flat[start : start + batch_size].to(device))
      chunks.append(_features_from_observations(actor, observations).cpu())
  return torch.cat(chunks)


def _paired_gate_window_dataset(
  off: dict[str, Any],
  rescued: torch.Tensor,
  *,
  environment_offset: int,
  pre_horizon: int,
  post_horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Label filter-off states by paired terminal rescue without future inputs."""
  data = off["dataset"]
  observations: list[torch.Tensor] = []
  labels: list[torch.Tensor] = []
  environment_ids: list[torch.Tensor] = []
  episode_summaries: list[dict[str, Any]] = []
  for environment_id in range(len(rescued)):
    rows = (data["environment_ids"] == environment_id).nonzero(
      as_tuple=False
    ).flatten()
    rows = rows[data["episode_steps"][rows].argsort()]
    interventions = data["would_intervene"][rows].nonzero(
      as_tuple=False
    ).flatten()
    if not len(interventions):
      continue
    first = int(interventions[0])
    start = max(0, first - int(pre_horizon))
    stop = min(len(rows), first + int(post_horizon) + 1)
    selected = rows[start:stop]
    if not len(selected):
      continue
    positive = bool(rescued[environment_id])
    observations.append(data["observations"][selected])
    labels.append(torch.full((len(selected),), positive, dtype=torch.bool))
    environment_ids.append(
      torch.full(
        (len(selected),),
        int(environment_offset + environment_id),
        dtype=torch.long,
      )
    )
    episode_summaries.append(
      {
        "environment_id": int(environment_offset + environment_id),
        "paired_rescue": positive,
        "first_intervention_step": first,
        "transition_count": len(selected),
      }
    )
  if not observations:
    raise RuntimeError("v113 collected no paired gate windows")
  dataset = {
    "observations": torch.cat(observations),
    "labels": torch.cat(labels),
    "environment_ids": torch.cat(environment_ids),
  }
  return dataset, {
    "episode_count": len(episode_summaries),
    "positive_episode_count": sum(row["paired_rescue"] for row in episode_summaries),
    "negative_episode_count": sum(not row["paired_rescue"] for row in episode_summaries),
    "transition_count": len(dataset["labels"]),
    "episodes": episode_summaries,
  }


def _gate_metrics(
  model: PairedTreatmentGatedResidual,
  features: torch.Tensor,
  labels: torch.Tensor,
  weights: torch.Tensor,
  *,
  threshold: float,
  device: str,
  batch_size: int,
) -> dict[str, float | int]:
  probability_chunks: list[torch.Tensor] = []
  model.eval()
  with torch.inference_mode():
    for start in range(0, len(features), batch_size):
      probability_chunks.append(
        model.gate_probability(features[start : start + batch_size].to(device)).cpu()
      )
  probabilities = torch.cat(probability_chunks)
  predicted = probabilities >= float(threshold)
  positive = labels
  negative = ~labels
  weighted_bce = F.binary_cross_entropy(
    probabilities.clamp(1.0e-6, 1.0 - 1.0e-6),
    labels.float(),
    weight=weights,
    reduction="sum",
  ) / weights.sum().clamp_min(1.0e-8)
  return {
    "transition_count": len(labels),
    "positive_transition_count": int(positive.sum()),
    "negative_transition_count": int(negative.sum()),
    "balanced_bce": float(weighted_bce),
    "positive_recall": float(predicted[positive].float().mean()),
    "negative_specificity": float((~predicted[negative]).float().mean()),
    "balanced_accuracy": 0.5
    * float(
      predicted[positive].float().mean()
      + (~predicted[negative]).float().mean()
    ),
    "positive_probability_mean": float(probabilities[positive].mean()),
    "negative_probability_mean": float(probabilities[negative].mean()),
  }


def _fit_gate(
  model: PairedTreatmentGatedResidual,
  features: torch.Tensor,
  labels: torch.Tensor,
  weights: torch.Tensor,
  *,
  threshold: float,
  learning_rate: float,
  epochs: int,
  batch_size: int,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  optimizer = torch.optim.Adam(model.gate.parameters(), lr=learning_rate)
  before = _gate_metrics(
    model,
    features,
    labels,
    weights,
    threshold=threshold,
    device=device,
    batch_size=batch_size,
  )
  updates = 0
  maximum_gradient_norm = 0.0
  model.train()
  for _ in range(epochs):
    permutation = torch.randperm(len(features))
    for start in range(0, len(permutation), batch_size):
      indices = permutation[start : start + batch_size]
      logits = model.gate(features[indices].to(device)).squeeze(-1)
      effective = weights[indices].to(device)
      per_loss = F.binary_cross_entropy_with_logits(
        logits, labels[indices].to(device).float(), reduction="none"
      )
      loss = (effective * per_loss).sum() / effective.sum().clamp_min(1.0e-8)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.gate.parameters(), max_grad_norm
      )
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v113 gate gradient is non-finite")
      optimizer.step()
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      updates += 1
  after = _gate_metrics(
    model,
    features,
    labels,
    weights,
    threshold=threshold,
    device=device,
    batch_size=batch_size,
  )
  return {
    "before": before,
    "after": after,
    "epochs": epochs,
    "optimizer_updates": updates,
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
  }, optimizer


def _residual_metrics(
  model: PairedTreatmentGatedResidual,
  features: torch.Tensor,
  targets: torch.Tensor,
  weights: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> dict[str, float | int]:
  distance_sum = loss_sum = prediction_norm_sum = 0.0
  weight_sum = 0.0
  model.eval()
  with torch.inference_mode():
    for start in range(0, len(features), batch_size):
      stop = min(start + batch_size, len(features))
      prediction = model.residual_action(features[start:stop].to(device))
      target = targets[start:stop].to(device)
      effective = weights[start:stop].to(device)
      distance = torch.linalg.vector_norm(prediction - target, dim=-1)
      per_loss = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=0.05
      ).mean(dim=-1)
      distance_sum += float((effective * distance).sum())
      loss_sum += float((effective * per_loss).sum())
      prediction_norm_sum += float(
        (effective * torch.linalg.vector_norm(prediction, dim=-1)).sum()
      )
      weight_sum += float(effective.sum())
  return {
    "transition_count": len(features),
    "weight_sum": weight_sum,
    "weighted_target_distance": distance_sum / max(1.0e-8, weight_sum),
    "weighted_smooth_l1": loss_sum / max(1.0e-8, weight_sum),
    "weighted_prediction_norm": prediction_norm_sum / max(1.0e-8, weight_sum),
  }


def _fit_residual(
  model: PairedTreatmentGatedResidual,
  features: torch.Tensor,
  targets: torch.Tensor,
  weights: torch.Tensor,
  trust_features: torch.Tensor,
  *,
  learning_rate: float,
  epochs: int,
  batch_size: int,
  trust_weight: float,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  optimizer = torch.optim.Adam(model.residual.parameters(), lr=learning_rate)
  before = _residual_metrics(
    model,
    features,
    targets,
    weights,
    device=device,
    batch_size=batch_size,
  )
  updates = 0
  maximum_gradient_norm = 0.0
  model.train()
  for _ in range(epochs):
    permutation = torch.randperm(len(features))
    for start in range(0, len(permutation), batch_size):
      indices = permutation[start : start + batch_size]
      prediction = model.residual_action(features[indices].to(device))
      target = targets[indices].to(device)
      effective = weights[indices].to(device)
      per_loss = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=0.05
      ).mean(dim=-1)
      teacher_loss = (effective * per_loss).sum() / effective.sum().clamp_min(
        1.0e-8
      )
      trust_count = min(len(indices), len(trust_features))
      trust_indices = torch.randint(len(trust_features), (trust_count,))
      trust_prediction = model.residual_action(
        trust_features[trust_indices].to(device)
      )
      trust_loss = F.smooth_l1_loss(
        trust_prediction, torch.zeros_like(trust_prediction), beta=0.05
      )
      loss = teacher_loss + float(trust_weight) * trust_loss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.residual.parameters(), max_grad_norm
      )
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v113 residual gradient is non-finite")
      optimizer.step()
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      updates += 1
  after = _residual_metrics(
    model,
    features,
    targets,
    weights,
    device=device,
    batch_size=batch_size,
  )
  return {
    "before": before,
    "after": after,
    "epochs": epochs,
    "optimizer_updates": updates,
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
  }, optimizer


def _policy_step(
  actor,
  model: PairedTreatmentGatedResidual,
  observations,
  *,
  threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  base_action = actor(observations, stochastic_output=False)
  features = _features_from_observations(actor, observations)
  probability = model.gate_probability(features)
  active = _geometry_active_from_observations(observations)
  gate = deployable_gate(probability, active, threshold=threshold)
  correction = gate.unsqueeze(-1) * model.residual_action(features)
  return base_action + correction, correction, gate, probability


def _evaluate_filter_off(
  runner,
  base_env,
  action_term,
  model: PairedTreatmentGatedResidual,
  *,
  seed: int,
  screen_envs: int,
  threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  action_term.set_runtime_filter_mask(
    torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
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
  considered = torch.arange(base_env.num_envs, device=base_env.device) < screen_envs
  active = considered.clone()
  success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  fell = torch.zeros_like(success)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  correction_norm_sum = gate_sum = probability_sum = 0.0
  transition_count = gate_active_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  runner.alg.actor.eval()
  model.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions, correction, gate, probability = _policy_step(
        runner.alg.actor, model, observations, threshold=threshold
      )
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        correction_norm_sum += float(
          torch.linalg.vector_norm(correction[ids], dim=-1).sum()
        )
        gate_sum += float(gate[ids].sum())
        probability_sum += float(probability[ids].sum())
        gate_active_count += int((gate[ids] > 0.0).sum())
        transition_count += len(ids)
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        reached_top = base_env.termination_manager.get_term("reached_top").bool()
        success[completed] = reached_top[completed]
        fell[completed] = extras["online_fell"][completed].bool()
        active &= ~completed
        if not bool(active.any()):
          observations = next_observations
          break
      observations = next_observations
  if bool(active.any()):
    raise RuntimeError("v113 did not finish the filter-off screen")
  selected = considered.nonzero(as_tuple=False).flatten().cpu()
  rows = [
    {
      "environment_id": int(environment_id),
      "success": bool(success[environment_id]),
      "fell": bool(fell[environment_id]),
      "steps": int(steps[environment_id]),
      "reached_risers": int(reached_risers[environment_id]),
    }
    for environment_id in selected
  ]
  success_count = int(success[considered].sum())
  return {
    "seed": seed,
    "num_episodes": screen_envs,
    "initial_state_signature": signature,
    "runtime_filter": False,
    "gate_threshold": threshold,
    "success_count": success_count,
    "success_rate": success_count / screen_envs,
    "fall_count": int(fell[considered].sum()),
    "fall_rate": float(fell[considered].float().mean()),
    "mean_reached_riser": float(reached_risers[considered].float().mean()),
    "mean_correction_norm": correction_norm_sum / max(1, transition_count),
    "mean_gate": gate_sum / max(1, transition_count),
    "mean_gate_probability": probability_sum / max(1, transition_count),
    "gate_active_fraction": gate_active_count / max(1, transition_count),
    "all_finite": all(
      math.isfinite(value)
      for value in (correction_norm_sum, gate_sum, probability_sum)
    ),
  }, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.training_seeds)
  if args.num_envs < 2 or not 1 <= args.screen_envs <= args.num_envs:
    raise ValueError("v113 environment counts are invalid")
  if args.gate_epochs < 1 or args.residual_epochs < 1 or args.batch_size < 1:
    raise ValueError("v113 optimization dimensions must be positive")
  if args.paired_pre_horizon < 0 or args.paired_post_horizon < 0:
    raise ValueError("v113 paired horizons must be non-negative")
  if not 0.0 < args.paired_pre_decay <= 1.0:
    raise ValueError("v113 paired decay must lie in (0, 1]")
  if not 0.0 < args.teacher_eta <= 1.0:
    raise ValueError("v113 teacher eta must lie in (0, 1]")
  if not 0.01 <= args.max_residual <= 1.0:
    raise ValueError("v113 residual bound must lie in [0.01, 1]")
  if not 0.0 < args.gate_threshold < 1.0:
    raise ValueError("v113 gate threshold must lie in (0, 1)")
  if min(args.gate_learning_rate, args.residual_learning_rate) <= 0.0:
    raise ValueError("v113 learning rates must be positive")
  if not 0.0 <= args.residual_trust_weight <= 4.0:
    raise ValueError("v113 residual trust weight must lie in [0, 4]")

  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v113 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v113 checkpoint or search protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v113 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v113 base checkpoint SHA-256 differs")
  output.mkdir(parents=True)
  started = time.monotonic()
  source_commit = _git(repo, "rev-parse", "HEAD")
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "screen_seed": args.screen_seed,
      "screen_envs": args.screen_envs,
      "gate_threshold": args.gate_threshold,
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
  geometry = configure_deployable_cbf_persistent_geometry_observation(env_cfg)
  env_cfg.scene.num_envs = max(args.num_envs, args.screen_envs)
  env_cfg.seed = seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = seeds[0]
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v113 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v113 requires the current velocity-CBF action")

  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    for parameter in runner.alg.actor.parameters():
      parameter.requires_grad_(False)
    model = PairedTreatmentGatedResidual(args.max_residual).to(args.device)
    initial_model_sha = actor_state_sha256(actor_state(model))

    gate_chunks: dict[str, list[torch.Tensor]] = {
      "observations": [],
      "labels": [],
      "environment_ids": [],
    }
    residual_chunks: dict[str, list[torch.Tensor]] = {
      "observations": [],
      "nominal_actions": [],
      "safe_actions": [],
      "would_intervene": [],
      "environment_ids": [],
    }
    residual_weight_chunks: list[torch.Tensor] = []
    trust_observation_chunks: list[torch.Tensor] = []
    rollout_summaries: list[dict[str, Any]] = []
    gate_window_summaries: list[dict[str, Any]] = []
    residual_trace_summaries: list[dict[str, Any]] = []
    global_environment_offset = 0
    rescued_count = harmed_count = 0

    for seed in seeds:
      off = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=False
      )
      on = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=True
      )
      if off["initial_state_signature"] != on["initial_state_signature"]:
        raise RuntimeError("v113 paired filter-on/off initial states differ")
      rescued = on["success"] & ~off["success"]
      harmed = off["success"] & ~on["success"]
      rescued_count += int(rescued.sum())
      harmed_count += int(harmed.sum())

      gate_dataset, gate_summary = _paired_gate_window_dataset(
        off,
        rescued,
        environment_offset=global_environment_offset,
        pre_horizon=args.paired_pre_horizon,
        post_horizon=args.paired_post_horizon,
      )
      for key in gate_chunks:
        gate_chunks[key].append(gate_dataset[key])
      gate_window_summaries.append({"seed": seed, **gate_summary})

      residual_dataset, residual_weights, residual_summary = (
        _paired_trajectory_rescue_dataset(
          off,
          on,
          rescued,
          environment_offset=global_environment_offset,
          pre_horizon=args.paired_pre_horizon,
          post_horizon=args.paired_post_horizon,
          pre_decay=args.paired_pre_decay,
          target_mode="deployment-counterfactual",
        )
      )
      if len(residual_weights):
        for key in residual_chunks:
          residual_chunks[key].append(residual_dataset[key])
        residual_weight_chunks.append(residual_weights)
      residual_trace_summaries.append({"seed": seed, **residual_summary})

      off_data = off["dataset"]
      off_success_transition = off["success"][off_data["environment_ids"]]
      trust_observation_chunks.append(
        off_data["observations"][off_success_transition]
      )
      rollout_summaries.extend(
        (
          {
            "seed": seed,
            "runtime_filter": False,
            "success_count": off["success_count"],
            "fall_count": off["fall_count"],
            "transition_count": len(off_data["observations"]),
          },
          {
            "seed": seed,
            "runtime_filter": True,
            "success_count": on["success_count"],
            "fall_count": on["fall_count"],
            "transition_count": len(on["dataset"]["observations"]),
            "rescued_episode_count": int(rescued.sum()),
            "harmed_episode_count": int(harmed.sum()),
          },
        )
      )
      print(json.dumps({"paired_rollout_completed": rollout_summaries[-2:]}), flush=True)
      global_environment_offset += base_env.num_envs

    if not residual_weight_chunks:
      raise RuntimeError("v113 collected no matched-rescue residual targets")
    gate_dataset = {key: torch.cat(chunks) for key, chunks in gate_chunks.items()}
    residual_dataset = {
      key: torch.cat(chunks) for key, chunks in residual_chunks.items()
    }
    residual_weights = torch.cat(residual_weight_chunks)
    trust_observations = torch.cat(
      [chunk for chunk in trust_observation_chunks if len(chunk)]
    )
    gate_weights = balanced_episode_weights(
      gate_dataset["labels"], gate_dataset["environment_ids"]
    )

    gate_features = _features_from_flat(
      runner.alg.actor,
      gate_dataset["observations"],
      device=args.device,
      batch_size=args.batch_size,
    )
    residual_features = _features_from_flat(
      runner.alg.actor,
      residual_dataset["observations"],
      device=args.device,
      batch_size=args.batch_size,
    )
    trust_features = _features_from_flat(
      runner.alg.actor,
      trust_observations,
      device=args.device,
      batch_size=args.batch_size,
    )
    residual_targets = torch.clamp(
      args.teacher_eta
      * (
        residual_dataset["safe_actions"]
        - residual_dataset["nominal_actions"]
      ),
      -args.max_residual,
      args.max_residual,
    )

    _seed_everything(args.optimization_seed)
    gate_training, gate_optimizer = _fit_gate(
      model,
      gate_features,
      gate_dataset["labels"],
      gate_weights,
      threshold=args.gate_threshold,
      learning_rate=args.gate_learning_rate,
      epochs=args.gate_epochs,
      batch_size=args.batch_size,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    _seed_everything(args.optimization_seed + 1)
    residual_training, residual_optimizer = _fit_residual(
      model,
      residual_features,
      residual_targets,
      residual_weights,
      trust_features,
      learning_rate=args.residual_learning_rate,
      epochs=args.residual_epochs,
      batch_size=args.batch_size,
      trust_weight=args.residual_trust_weight,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )

    final_model_state = actor_state(model)
    final_model_sha = actor_state_sha256(final_model_state)
    candidate_path = output / "candidate.pt"
    _atomic_torch(
      candidate_path,
      {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "git_commit": source_commit,
        "base_checkpoint_sha256": checkpoint_sha,
        "base_actor_state_dict": {
          key: value.detach().cpu() for key, value in expanded_state.items()
        },
        "base_actor_sha256": actor_state_sha256(expanded_state),
        "gated_residual_state_dict": {
          key: value.detach().cpu() for key, value in final_model_state.items()
        },
        "gated_residual_sha256": final_model_sha,
        "model_config": {
          "feature_dim": FEATURE_DIM,
          "base_hidden_dim": BASE_HIDDEN_DIM,
          "geometry_dim": PERSISTENT_GEOMETRY_OBSERVATION_DIM,
          "action_dim": ACTION_DIM,
          "gate_hidden_dims": [64, 32],
          "residual_hidden_dims": [128, 64],
          "max_residual": args.max_residual,
          "gate_threshold": args.gate_threshold,
        },
      },
    )
    screen, screen_rows = _evaluate_filter_off(
      runner,
      base_env,
      action_term,
      model,
      seed=args.screen_seed,
      screen_envs=args.screen_envs,
      threshold=args.gate_threshold,
    )
    screen["candidate_checkpoint_sha256"] = file_sha256(candidate_path)
    screen["gated_residual_sha256"] = final_model_sha
    screen["passed_75_percent"] = screen["success_rate"] >= 0.75
    _atomic_json(output / "screen_summary.json", screen)
    _write_csv(output / "screen_episodes.csv", screen_rows)

    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "context": args.context,
      "base_checkpoint_sha256": checkpoint_sha,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "base_actor_sha256": actor_state_sha256(expanded_state),
      "initial_gated_residual_sha256": initial_model_sha,
      "final_gated_residual_sha256": final_model_sha,
      "trainable_parameter_count": sum(
        parameter.numel() for parameter in model.parameters()
      ),
      "actor_observation_dim": (
        LEGACY_ACTOR_OBSERVATION_DIM + PERSISTENT_GEOMETRY_OBSERVATION_DIM
      ),
      "training_seeds": seeds,
      "num_envs": base_env.num_envs,
      "optimization_seed": args.optimization_seed,
      "paired_trace": {
        "pre_horizon": args.paired_pre_horizon,
        "post_horizon": args.paired_post_horizon,
        "pre_decay": args.paired_pre_decay,
      },
      "teacher_eta": args.teacher_eta,
      "max_residual": args.max_residual,
      "gate_threshold": args.gate_threshold,
      "rescued_episode_count": rescued_count,
      "harmed_episode_count": harmed_count,
      "gate_transition_count": len(gate_features),
      "residual_transition_count": len(residual_features),
      "trust_transition_count": len(trust_features),
      "rollout_summaries": rollout_summaries,
      "gate_window_summaries": gate_window_summaries,
      "residual_trace_summaries": residual_trace_summaries,
      "gate_training": gate_training,
      "residual_training": residual_training,
      "screen": screen,
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "optimizer_state_in_candidate": False,
      "gate_optimizer_updates": len(gate_optimizer.state),
      "residual_optimizer_updates": len(residual_optimizer.state),
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
