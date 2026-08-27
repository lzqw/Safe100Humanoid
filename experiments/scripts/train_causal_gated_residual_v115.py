"""Train a causal recurrent gate on paired discordant CBF outcomes.

v113 used a memoryless classifier and labeled every non-rescue as negative.
That local classifier barely separated the classes.  v115 restricts the value
question to genuinely discordant pairs (CBF rescued versus CBF harmed) and uses
a GRU over the deployable filter-off history.  The GRU is causal: its decision
at step t cannot access observations after t.  A bounded residual is trained
only on rescued trajectories and is enabled above a conservative confidence
threshold; otherwise deployment is exactly the frozen v79 actor.
"""

from __future__ import annotations

import argparse
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
from train_paired_gated_residual_v113 import (
  ACTION_DIM,
  BASE_HIDDEN_DIM,
  FEATURE_DIM,
  _features_from_flat,
  _features_from_observations,
  _fit_residual,
  _geometry_active_from_observations,
  deployable_gate,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "causal-discordant-treatment-gated-cbf-residual-v115"
GATE_HIDDEN_DIM = 64


class CausalGatedResidual(torch.nn.Module):
  """Causal GRU treatment gate plus an independent bounded residual head."""

  def __init__(
    self,
    max_residual: float,
    gate_hidden_dim: int = GATE_HIDDEN_DIM,
    residual_hidden_dims: tuple[int, int] = (128, 64),
  ) -> None:
    super().__init__()
    self.max_residual = float(max_residual)
    self.gate_hidden_dim = int(gate_hidden_dim)
    self.gate_gru = torch.nn.GRU(
      FEATURE_DIM, self.gate_hidden_dim, batch_first=True
    )
    self.gate_head = torch.nn.Linear(self.gate_hidden_dim, 1)
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

  def gate_logits_sequence(
    self,
    padded_features: torch.Tensor,
    lengths: torch.Tensor,
  ) -> torch.Tensor:
    packed = torch.nn.utils.rnn.pack_padded_sequence(
      padded_features,
      lengths.detach().cpu(),
      batch_first=True,
      enforce_sorted=False,
    )
    packed_output, _ = self.gate_gru(packed)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(
      packed_output,
      batch_first=True,
      total_length=padded_features.shape[1],
    )
    return self.gate_head(output).squeeze(-1)

  def gate_step(
    self,
    features: torch.Tensor,
    hidden: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    output, next_hidden = self.gate_gru(features.unsqueeze(1), hidden)
    probability = torch.sigmoid(self.gate_head(output[:, 0]).squeeze(-1))
    return probability, next_hidden

  def residual_action(self, features: torch.Tensor) -> torch.Tensor:
    raw = self.residual(features)
    return self.max_residual * torch.tanh(raw / self.max_residual)


def balanced_sequence_weights(
  labels: torch.Tensor,
  loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
  """Balance rescue/harm classes and give every episode equal class mass."""
  if (
    labels.ndim != 1
    or labels.dtype != torch.bool
    or len(labels) != len(loss_masks)
    or not len(labels)
  ):
    raise ValueError("v115 sequence labels or masks are invalid")
  positive_count = int(labels.sum())
  negative_count = len(labels) - positive_count
  if positive_count == 0 or negative_count == 0:
    raise ValueError("v115 requires both rescued and harmed sequences")
  weights: list[torch.Tensor] = []
  for label, mask in zip(labels.tolist(), loss_masks):
    if mask.ndim != 1 or mask.dtype != torch.bool or not bool(mask.any()):
      raise ValueError("v115 sequence loss mask is invalid")
    class_count = positive_count if label else negative_count
    value = 0.5 / float(class_count * int(mask.sum()))
    weights.append(mask.float() * value)
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
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--screen-seed", type=int, required=True)
  parser.add_argument("--screen-envs", type=int, default=64)
  parser.add_argument("--paired-pre-horizon", type=int, default=20)
  parser.add_argument("--paired-post-horizon", type=int, default=50)
  parser.add_argument("--paired-pre-decay", type=float, default=0.9)
  parser.add_argument("--teacher-eta", type=float, default=0.25)
  parser.add_argument("--max-residual", type=float, default=0.25)
  parser.add_argument("--gate-threshold", type=float, default=0.6)
  parser.add_argument("--gate-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--gate-epochs", type=int, default=50)
  parser.add_argument("--residual-epochs", type=int, default=20)
  parser.add_argument("--gate-batch-episodes", type=int, default=32)
  parser.add_argument("--residual-batch-size", type=int, default=4096)
  parser.add_argument("--residual-trust-weight", type=float, default=0.25)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v115 seeds must be comma-separated integers") from exc
  if len(seeds) < 4 or len(set(seeds)) != len(seeds):
    raise ValueError("v115 requires at least four unique paired seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v115 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _paired_gate_sequences(
  off: dict[str, Any],
  rescued: torch.Tensor,
  harmed: torch.Tensor,
  *,
  pre_horizon: int,
  post_horizon: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, dict[str, Any]]:
  """Build causal prefixes and loss windows for discordant episodes only."""
  if rescued.shape != harmed.shape or bool((rescued & harmed).any()):
    raise ValueError("v115 paired outcome masks are incompatible")
  data = off["dataset"]
  observations: list[torch.Tensor] = []
  loss_masks: list[torch.Tensor] = []
  labels: list[bool] = []
  episode_summaries: list[dict[str, Any]] = []
  selected = rescued | harmed
  for environment_id in selected.nonzero(as_tuple=False).flatten().tolist():
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
    stop = min(len(rows), first + int(post_horizon) + 1)
    start_loss = max(0, first - int(pre_horizon))
    sequence_rows = rows[:stop]
    mask = torch.zeros(len(sequence_rows), dtype=torch.bool)
    mask[start_loss:] = True
    positive = bool(rescued[environment_id])
    observations.append(data["observations"][sequence_rows])
    loss_masks.append(mask)
    labels.append(positive)
    episode_summaries.append(
      {
        "environment_id": int(environment_id),
        "paired_outcome": "rescued" if positive else "harmed",
        "first_intervention_step": first,
        "sequence_length": len(sequence_rows),
        "loss_transition_count": int(mask.sum()),
      }
    )
  if not observations:
    raise RuntimeError("v115 collected no discordant causal sequences")
  label_tensor = torch.tensor(labels, dtype=torch.bool)
  return observations, loss_masks, label_tensor, {
    "episode_count": len(labels),
    "rescued_episode_count": int(label_tensor.sum()),
    "harmed_episode_count": int((~label_tensor).sum()),
    "sequence_transition_count": sum(len(value) for value in observations),
    "loss_transition_count": sum(int(value.sum()) for value in loss_masks),
    "episodes": episode_summaries,
  }


def _featurize_sequences(
  actor,
  observations: list[torch.Tensor],
  *,
  device: str,
  batch_size: int,
) -> list[torch.Tensor]:
  lengths = [len(value) for value in observations]
  flat_features = _features_from_flat(
    actor,
    torch.cat(observations),
    device=device,
    batch_size=batch_size,
  )
  return list(torch.split(flat_features, lengths))


def _gate_metrics(
  model: CausalGatedResidual,
  sequences: list[torch.Tensor],
  labels: torch.Tensor,
  weights: list[torch.Tensor],
  *,
  threshold: float,
  batch_episodes: int,
  device: str,
) -> dict[str, float | int]:
  probability_sequences: list[torch.Tensor] = []
  model.eval()
  with torch.inference_mode():
    for start in range(0, len(sequences), batch_episodes):
      batch = sequences[start : start + batch_episodes]
      lengths = torch.tensor([len(value) for value in batch], dtype=torch.long)
      padded = torch.nn.utils.rnn.pad_sequence(
        [value.to(device) for value in batch], batch_first=True
      )
      logits = model.gate_logits_sequence(padded, lengths)
      for row, length in enumerate(lengths.tolist()):
        probability_sequences.append(torch.sigmoid(logits[row, :length]).cpu())
  state_probabilities = torch.cat(
    [probability[weight > 0.0] for probability, weight in zip(probability_sequences, weights)]
  )
  state_labels = torch.cat(
    [
      torch.full((int((weight > 0.0).sum()),), bool(label), dtype=torch.bool)
      for label, weight in zip(labels.tolist(), weights)
    ]
  )
  state_weights = torch.cat([weight[weight > 0.0] for weight in weights])
  episode_probabilities = torch.tensor(
    [
      float((probability * weight).sum() / weight.sum().clamp_min(1.0e-8))
      for probability, weight in zip(probability_sequences, weights)
    ]
  )
  state_predicted = state_probabilities >= float(threshold)
  episode_predicted = episode_probabilities >= float(threshold)
  positive_state = state_labels
  negative_state = ~state_labels
  positive_episode = labels
  negative_episode = ~labels
  weighted_bce = F.binary_cross_entropy(
    state_probabilities.clamp(1.0e-6, 1.0 - 1.0e-6),
    state_labels.float(),
    weight=state_weights,
    reduction="sum",
  ) / state_weights.sum().clamp_min(1.0e-8)
  return {
    "episode_count": len(labels),
    "rescued_episode_count": int(positive_episode.sum()),
    "harmed_episode_count": int(negative_episode.sum()),
    "loss_transition_count": len(state_labels),
    "balanced_bce": float(weighted_bce),
    "state_positive_recall": float(
      state_predicted[positive_state].float().mean()
    ),
    "state_negative_specificity": float(
      (~state_predicted[negative_state]).float().mean()
    ),
    "state_balanced_accuracy": 0.5
    * float(
      state_predicted[positive_state].float().mean()
      + (~state_predicted[negative_state]).float().mean()
    ),
    "episode_positive_recall": float(
      episode_predicted[positive_episode].float().mean()
    ),
    "episode_negative_specificity": float(
      (~episode_predicted[negative_episode]).float().mean()
    ),
    "episode_balanced_accuracy": 0.5
    * float(
      episode_predicted[positive_episode].float().mean()
      + (~episode_predicted[negative_episode]).float().mean()
    ),
    "rescued_probability_mean": float(
      episode_probabilities[positive_episode].mean()
    ),
    "harmed_probability_mean": float(
      episode_probabilities[negative_episode].mean()
    ),
  }


def _fit_gate(
  model: CausalGatedResidual,
  train_sequences: list[torch.Tensor],
  train_labels: torch.Tensor,
  train_weights: list[torch.Tensor],
  validation_sequences: list[torch.Tensor],
  validation_labels: torch.Tensor,
  validation_weights: list[torch.Tensor],
  *,
  threshold: float,
  learning_rate: float,
  epochs: int,
  batch_episodes: int,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  parameters = list(model.gate_gru.parameters()) + list(model.gate_head.parameters())
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  before_train = _gate_metrics(
    model,
    train_sequences,
    train_labels,
    train_weights,
    threshold=threshold,
    batch_episodes=batch_episodes,
    device=device,
  )
  before_validation = _gate_metrics(
    model,
    validation_sequences,
    validation_labels,
    validation_weights,
    threshold=threshold,
    batch_episodes=batch_episodes,
    device=device,
  )
  updates = 0
  maximum_gradient_norm = 0.0
  model.train()
  for _ in range(epochs):
    permutation = torch.randperm(len(train_sequences))
    for start in range(0, len(permutation), batch_episodes):
      indices = permutation[start : start + batch_episodes].tolist()
      batch_sequences = [train_sequences[index].to(device) for index in indices]
      batch_weights = [train_weights[index].to(device) for index in indices]
      lengths = torch.tensor(
        [len(value) for value in batch_sequences], dtype=torch.long
      )
      padded = torch.nn.utils.rnn.pad_sequence(batch_sequences, batch_first=True)
      padded_weights = torch.nn.utils.rnn.pad_sequence(
        batch_weights, batch_first=True
      )
      logits = model.gate_logits_sequence(padded, lengths)
      targets = train_labels[indices].to(device).float().unsqueeze(-1).expand_as(logits)
      per_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
      )
      loss = (padded_weights * per_loss).sum() / padded_weights.sum().clamp_min(
        1.0e-8
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v115 causal gate gradient is non-finite")
      optimizer.step()
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      updates += 1
  after_train = _gate_metrics(
    model,
    train_sequences,
    train_labels,
    train_weights,
    threshold=threshold,
    batch_episodes=batch_episodes,
    device=device,
  )
  after_validation = _gate_metrics(
    model,
    validation_sequences,
    validation_labels,
    validation_weights,
    threshold=threshold,
    batch_episodes=batch_episodes,
    device=device,
  )
  return {
    "before": {"train": before_train, "validation": before_validation},
    "after": {"train": after_train, "validation": after_validation},
    "epochs": epochs,
    "optimizer_updates": updates,
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
  }, optimizer


def _policy_step(
  actor,
  model: CausalGatedResidual,
  observations,
  hidden: torch.Tensor,
  *,
  threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  base_action = actor(observations, stochastic_output=False)
  features = _features_from_observations(actor, observations)
  probability, next_hidden = model.gate_step(features, hidden)
  active = _geometry_active_from_observations(observations)
  gate = deployable_gate(probability, active, threshold=threshold)
  correction = gate.unsqueeze(-1) * model.residual_action(features)
  return base_action + correction, correction, gate, probability, next_hidden


def _evaluate_filter_off(
  runner,
  base_env,
  action_term,
  model: CausalGatedResidual,
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
  hidden = torch.zeros(
    (1, base_env.num_envs, model.gate_hidden_dim),
    dtype=torch.float32,
    device=base_env.device,
  )
  correction_norm_sum = gate_sum = probability_sum = 0.0
  transition_count = gate_active_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  runner.alg.actor.eval()
  model.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions, correction, gate, probability, hidden = _policy_step(
        runner.alg.actor,
        model,
        observations,
        hidden,
        threshold=threshold,
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
    raise RuntimeError("v115 did not finish the filter-off screen")
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
    "passed_75_percent": success_count / screen_envs >= 0.75,
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
  validation_seed = seeds[-1]
  if args.num_envs < 2 or not 1 <= args.screen_envs <= args.num_envs:
    raise ValueError("v115 environment counts are invalid")
  if min(args.gate_epochs, args.residual_epochs, args.gate_batch_episodes) < 1:
    raise ValueError("v115 optimization dimensions must be positive")
  if args.residual_batch_size < 1 or args.paired_pre_horizon < 0:
    raise ValueError("v115 batch size or paired horizon is invalid")
  if args.paired_post_horizon < 0 or not 0.0 < args.paired_pre_decay <= 1.0:
    raise ValueError("v115 paired trace configuration is invalid")
  if not 0.0 < args.teacher_eta <= 1.0 or not 0.01 <= args.max_residual <= 1.0:
    raise ValueError("v115 residual target configuration is invalid")
  if not 0.5 <= args.gate_threshold < 1.0:
    raise ValueError("v115 gate threshold must lie in [0.5, 1)")
  if min(args.gate_learning_rate, args.residual_learning_rate) <= 0.0:
    raise ValueError("v115 learning rates must be positive")

  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v115 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v115 checkpoint or search protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v115 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v115 base checkpoint SHA-256 differs")
  output.mkdir(parents=True)
  started = time.monotonic()
  source_commit = _git(repo, "rev-parse", "HEAD")
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": seeds[:-1],
      "validation_seed": validation_seed,
      "num_envs": args.num_envs,
      "screen_seed": args.screen_seed,
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
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = seeds[0]
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v115 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v115 requires the current velocity-CBF action")

  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    for parameter in runner.alg.actor.parameters():
      parameter.requires_grad_(False)
    model = CausalGatedResidual(args.max_residual).to(args.device)
    initial_model_sha = actor_state_sha256(actor_state(model))

    train_sequence_observations: list[torch.Tensor] = []
    train_sequence_masks: list[torch.Tensor] = []
    train_sequence_labels: list[torch.Tensor] = []
    validation_sequence_observations: list[torch.Tensor] = []
    validation_sequence_masks: list[torch.Tensor] = []
    validation_sequence_labels: list[torch.Tensor] = []
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
    sequence_summaries: list[dict[str, Any]] = []
    residual_trace_summaries: list[dict[str, Any]] = []
    global_environment_offset = 0

    for seed in seeds:
      off = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=False
      )
      on = _collect_first_episodes(
        runner, base_env, action_term, seed=seed, runtime_filter=True
      )
      if off["initial_state_signature"] != on["initial_state_signature"]:
        raise RuntimeError("v115 paired filter-on/off initial states differ")
      rescued = on["success"] & ~off["success"]
      harmed = off["success"] & ~on["success"]
      observations, masks, labels, sequence_summary = _paired_gate_sequences(
        off,
        rescued,
        harmed,
        pre_horizon=args.paired_pre_horizon,
        post_horizon=args.paired_post_horizon,
      )
      sequence_summaries.append({"seed": seed, **sequence_summary})
      if seed == validation_seed:
        validation_sequence_observations.extend(observations)
        validation_sequence_masks.extend(masks)
        validation_sequence_labels.append(labels)
      else:
        train_sequence_observations.extend(observations)
        train_sequence_masks.extend(masks)
        train_sequence_labels.append(labels)
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
            "split": "validation" if seed == validation_seed else "train",
            "runtime_filter": False,
            "success_count": off["success_count"],
            "fall_count": off["fall_count"],
          },
          {
            "seed": seed,
            "split": "validation" if seed == validation_seed else "train",
            "runtime_filter": True,
            "success_count": on["success_count"],
            "fall_count": on["fall_count"],
            "rescued_episode_count": int(rescued.sum()),
            "harmed_episode_count": int(harmed.sum()),
          },
        )
      )
      print(json.dumps({"paired_rollout_completed": rollout_summaries[-2:]}), flush=True)
      global_environment_offset += args.num_envs

    train_labels = torch.cat(train_sequence_labels)
    validation_labels = torch.cat(validation_sequence_labels)
    train_weights = balanced_sequence_weights(
      train_labels, train_sequence_masks
    )
    validation_weights = balanced_sequence_weights(
      validation_labels, validation_sequence_masks
    )
    train_sequences = _featurize_sequences(
      runner.alg.actor,
      train_sequence_observations,
      device=args.device,
      batch_size=args.residual_batch_size,
    )
    validation_sequences = _featurize_sequences(
      runner.alg.actor,
      validation_sequence_observations,
      device=args.device,
      batch_size=args.residual_batch_size,
    )
    residual_dataset = {
      key: torch.cat(chunks) for key, chunks in residual_chunks.items()
    }
    residual_weights = torch.cat(residual_weight_chunks)
    trust_observations = torch.cat(trust_observation_chunks)
    residual_features = _features_from_flat(
      runner.alg.actor,
      residual_dataset["observations"],
      device=args.device,
      batch_size=args.residual_batch_size,
    )
    trust_features = _features_from_flat(
      runner.alg.actor,
      trust_observations,
      device=args.device,
      batch_size=args.residual_batch_size,
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
      train_sequences,
      train_labels,
      train_weights,
      validation_sequences,
      validation_labels,
      validation_weights,
      threshold=args.gate_threshold,
      learning_rate=args.gate_learning_rate,
      epochs=args.gate_epochs,
      batch_episodes=args.gate_batch_episodes,
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
      batch_size=args.residual_batch_size,
      trust_weight=args.residual_trust_weight,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )

    final_state = actor_state(model)
    final_model_sha = actor_state_sha256(final_state)
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
        "causal_gated_residual_state_dict": {
          key: value.detach().cpu() for key, value in final_state.items()
        },
        "causal_gated_residual_sha256": final_model_sha,
        "model_config": {
          "feature_dim": FEATURE_DIM,
          "base_hidden_dim": BASE_HIDDEN_DIM,
          "geometry_dim": PERSISTENT_GEOMETRY_OBSERVATION_DIM,
          "action_dim": ACTION_DIM,
          "gate_hidden_dim": GATE_HIDDEN_DIM,
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
    screen["causal_gated_residual_sha256"] = final_model_sha
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
      "initial_causal_gated_residual_sha256": initial_model_sha,
      "final_causal_gated_residual_sha256": final_model_sha,
      "trainable_parameter_count": sum(
        parameter.numel() for parameter in model.parameters()
      ),
      "actor_observation_dim": (
        LEGACY_ACTOR_OBSERVATION_DIM + PERSISTENT_GEOMETRY_OBSERVATION_DIM
      ),
      "training_seeds": seeds[:-1],
      "validation_seed": validation_seed,
      "num_envs": args.num_envs,
      "optimization_seed": args.optimization_seed,
      "paired_trace": {
        "pre_horizon": args.paired_pre_horizon,
        "post_horizon": args.paired_post_horizon,
        "pre_decay": args.paired_pre_decay,
      },
      "teacher_eta": args.teacher_eta,
      "max_residual": args.max_residual,
      "gate_threshold": args.gate_threshold,
      "rollout_summaries": rollout_summaries,
      "sequence_summaries": sequence_summaries,
      "residual_trace_summaries": residual_trace_summaries,
      "gate_training": gate_training,
      "residual_training": residual_training,
      "screen": screen,
      "independent_gate_run": False,
      "independent_gate_policy": "run separately only if screen_rate_gte_0.75",
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "optimizer_state_in_candidate": false,
      "gate_optimizer_parameter_state_count": len(gate_optimizer.state),
      "residual_optimizer_parameter_state_count": len(residual_optimizer.state),
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
