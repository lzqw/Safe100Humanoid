"""Optimize a deployable residual directly on filter-off episode outcomes.

All prior distillation variants learned an action supplied by a CBF trajectory.
v117 instead freezes the v79 actor and collects stochastic residual actions in
the exact filter-off deployment state distribution.  Successful and failed
episodes receive balanced positive/negative advantages, and one clipped PPO
update trains only the small residual head.  The update is projected to a fixed
forward-KL cap before a deterministic filter-off screen.
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
from cbf_teacher_v31_protocol import (
  CLEARANCE_BARRIER_SLOPE,
  CONTEXTS,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_observable_cbf_adapter_v49 import _expand_actor_state
from refine_observable_cbf_ppo_v51 import _expand_critic_state
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from train_learned_residual_v97 import LearnedCbfResidual
from train_paired_gated_residual_v113 import (
  _features_from_observations,
  _geometry_active_from_observations,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "filter-off-episodic-residual-ppo-v117"
FULL_BATCH_METHOD_ID = "filter-off-episodic-full-batch-residual-ppo-v118"
VALIDATED_FULL_BATCH_METHOD_ID = (
  "heldout-validated-filter-off-full-batch-residual-ppo-v119"
)
PROGRESS_VALIDATED_METHOD_ID = (
  "heldout-validated-terminal-progress-residual-ppo-v120"
)
CRITIC_GAE_VALIDATED_METHOD_ID = (
  "heldout-validated-pretrained-critic-gae-residual-ppo-v121"
)
ANTITHETIC_VALIDATED_METHOD_ID = (
  "heldout-validated-paired-antithetic-residual-ppo-v122"
)
CALIBRATED_ANTITHETIC_METHOD_ID = (
  "heldout-calibrated-paired-antithetic-residual-ppo-v123"
)
JOINT_SCALE_GRID = tuple(0.5**index for index in range(8))
MINIMUM_JOINT_SURROGATE_GAIN = 1.0e-6
ACTION_DIM = 12


def _build_optimizer(
  residual: LearnedCbfResidual,
  *,
  name: str,
  learning_rate: float,
) -> torch.optim.Optimizer:
  if name == "adam":
    return torch.optim.Adam(residual.parameters(), lr=learning_rate)
  if name == "sgd":
    return torch.optim.SGD(residual.parameters(), lr=learning_rate)
  raise ValueError("v117/v118 residual optimizer is unsupported")


def balanced_outcome_weights(
  environment_ids: torch.Tensor,
  episode_success: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Balance outcome classes and give every episode equal class mass."""
  if (
    environment_ids.ndim != 1
    or environment_ids.dtype != torch.long
    or episode_success.ndim != 1
    or episode_success.dtype != torch.bool
    or not len(environment_ids)
    or int(environment_ids.max()) >= len(episode_success)
  ):
    raise ValueError("v117 outcome ids or labels are invalid")
  present = environment_ids.unique(sorted=True)
  present_success = episode_success[present]
  positive_count = int(present_success.sum())
  negative_count = len(present) - positive_count
  if positive_count == 0 or negative_count == 0:
    raise ValueError("v117 requires successful and failed explored episodes")
  weights = torch.zeros(len(environment_ids), dtype=torch.float32)
  advantages = torch.empty(len(environment_ids), dtype=torch.float32)
  for environment_id in present.tolist():
    rows = (environment_ids == int(environment_id)).nonzero(
      as_tuple=False
    ).flatten()
    positive = bool(episode_success[environment_id])
    class_count = positive_count if positive else negative_count
    weights[rows] = 0.5 / float(class_count * len(rows))
    advantages[rows] = 1.0 if positive else -1.0
  return weights, advantages


def standardized_progress_weights(
  environment_ids: torch.Tensor,
  episode_success: torch.Tensor,
  episode_reached_risers: torch.Tensor,
  *,
  num_envs: int,
  success_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Give each episode equal mass and standardize terminal progress per seed."""
  if (
    environment_ids.ndim != 1
    or environment_ids.dtype != torch.long
    or episode_success.ndim != 1
    or episode_success.dtype != torch.bool
    or episode_reached_risers.shape != episode_success.shape
    or num_envs < 1
    or success_bonus < 0.0
    or not len(environment_ids)
  ):
    raise ValueError("v120 terminal-progress inputs are invalid")
  present = environment_ids.unique(sorted=True)
  if int(present.max()) >= len(episode_success):
    raise ValueError("v120 terminal-progress episode id is out of range")
  seed_ids = torch.div(present, int(num_envs), rounding_mode="floor")
  unique_seeds = seed_ids.unique(sorted=True)
  weights = torch.zeros(len(environment_ids), dtype=torch.float32)
  advantages = torch.zeros(len(environment_ids), dtype=torch.float32)
  for seed_id in unique_seeds.tolist():
    seed_episodes = present[seed_ids == int(seed_id)]
    scores = episode_reached_risers[seed_episodes].float() + float(
      success_bonus
    ) * episode_success[seed_episodes].float()
    normalized = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(
      1.0e-6
    )
    for environment_id, advantage in zip(seed_episodes.tolist(), normalized.tolist()):
      rows = (environment_ids == int(environment_id)).nonzero(
        as_tuple=False
      ).flatten()
      weights[rows] = 1.0 / float(
        len(unique_seeds) * len(seed_episodes) * len(rows)
      )
      advantages[rows] = float(advantage)
  return weights, advantages


def compute_pretrained_critic_gae(
  critic_dataset: dict[str, torch.Tensor],
  *,
  gamma: float,
  gae_lambda: float,
) -> torch.Tensor:
  """Compute causal GAE on complete first episodes using stored critic values."""
  required = {"values", "rewards", "dones", "environment_ids", "episode_steps"}
  if (
    set(critic_dataset) != required
    or not 0.0 < gamma <= 1.0
    or not 0.0 <= gae_lambda <= 1.0
  ):
    raise ValueError("v121 critic dataset or GAE parameters are invalid")
  count = len(critic_dataset["values"])
  if (
    count == 0
    or any(value.ndim != 1 or len(value) != count for value in critic_dataset.values())
    or critic_dataset["dones"].dtype != torch.bool
    or critic_dataset["environment_ids"].dtype != torch.long
    or critic_dataset["episode_steps"].dtype != torch.long
    or not bool(torch.isfinite(critic_dataset["values"]).all())
    or not bool(torch.isfinite(critic_dataset["rewards"]).all())
  ):
    raise ValueError("v121 critic transition tensors are empty or misaligned")
  advantages = torch.zeros(count, dtype=torch.float32)
  for environment_id in critic_dataset["environment_ids"].unique(sorted=True).tolist():
    rows = (critic_dataset["environment_ids"] == int(environment_id)).nonzero(
      as_tuple=False
    ).flatten()
    rows = rows[critic_dataset["episode_steps"][rows].argsort()]
    episode_steps = critic_dataset["episode_steps"][rows]
    expected_steps = torch.arange(len(rows), dtype=torch.long)
    episode_dones = critic_dataset["dones"][rows]
    if (
      not torch.equal(episode_steps, expected_steps)
      or not bool(episode_dones[-1])
      or bool(episode_dones[:-1].any())
    ):
      raise ValueError("v121 requires one complete ordered first episode per env")
    running = 0.0
    for position in range(len(rows) - 1, -1, -1):
      row = int(rows[position])
      done = bool(critic_dataset["dones"][row])
      next_value = (
        0.0
        if done or position + 1 == len(rows)
        else float(critic_dataset["values"][int(rows[position + 1])])
      )
      nonterminal = 0.0 if done else 1.0
      delta = (
        float(critic_dataset["rewards"][row])
        + float(gamma) * next_value * nonterminal
        - float(critic_dataset["values"][row])
      )
      running = delta + float(gamma) * float(gae_lambda) * nonterminal * running
      advantages[row] = running
  return advantages


def standardized_gae_weights(
  environment_ids: torch.Tensor,
  full_transition_indices: torch.Tensor,
  full_advantages: torch.Tensor,
  *,
  num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Normalize GAE per rollout seed while keeping every episode equally weighted."""
  if (
    environment_ids.shape != full_transition_indices.shape
    or environment_ids.dtype != torch.long
    or full_transition_indices.dtype != torch.long
    or not len(environment_ids)
    or int(full_transition_indices.max()) >= len(full_advantages)
    or num_envs < 1
  ):
    raise ValueError("v121 actor/critic transition mapping is invalid")
  raw = full_advantages[full_transition_indices].float()
  advantages = torch.zeros_like(raw)
  seed_ids = torch.div(environment_ids, int(num_envs), rounding_mode="floor")
  unique_seeds = seed_ids.unique(sorted=True)
  weights = torch.zeros(len(environment_ids), dtype=torch.float32)
  for seed_id in unique_seeds.tolist():
    seed_rows = (seed_ids == int(seed_id)).nonzero(as_tuple=False).flatten()
    seed_values = raw[seed_rows]
    advantages[seed_rows] = (seed_values - seed_values.mean()) / seed_values.std(
      unbiased=False
    ).clamp_min(1.0e-6)
    seed_episodes = environment_ids[seed_rows].unique(sorted=True)
    for environment_id in seed_episodes.tolist():
      episode_rows = (environment_ids == int(environment_id)).nonzero(
        as_tuple=False
      ).flatten()
      weights[episode_rows] = 1.0 / float(
        len(unique_seeds) * len(seed_episodes) * len(episode_rows)
      )
  return weights, advantages


def standardized_antithetic_weights(
  pair_ids: torch.Tensor,
  branch_signs: torch.Tensor,
  pair_score_differences: torch.Tensor,
  *,
  num_pairs_per_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Center matched +/- trajectory differences and weight every branch equally."""
  if (
    pair_ids.ndim != 1
    or pair_ids.dtype != torch.long
    or branch_signs.shape != pair_ids.shape
    or branch_signs.dtype != torch.int8
    or pair_score_differences.ndim != 1
    or not len(pair_ids)
    or int(pair_ids.min()) < 0
    or int(pair_ids.max()) >= len(pair_score_differences)
    or not bool(torch.isfinite(pair_score_differences).all())
    or num_pairs_per_seed < 2
  ):
    raise ValueError("v122 paired antithetic inputs are invalid")
  weights = torch.zeros(len(pair_ids), dtype=torch.float32)
  advantages = torch.zeros(len(pair_ids), dtype=torch.float32)
  seed_ids = torch.div(pair_ids, int(num_pairs_per_seed), rounding_mode="floor")
  unique_seeds = seed_ids.unique(sorted=True)
  for seed_id in unique_seeds.tolist():
    seed_rows = (seed_ids == int(seed_id)).nonzero(as_tuple=False).flatten()
    seed_pairs = pair_ids[seed_rows].unique(sorted=True)
    differences = pair_score_differences[seed_pairs].float()
    normalized = (differences - differences.mean()) / differences.std(
      unbiased=False
    ).clamp_min(1.0e-6)
    for pair_id, pair_advantage in zip(seed_pairs.tolist(), normalized.tolist()):
      for branch_sign in (-1, 1):
        branch_rows = (
          (pair_ids == int(pair_id)) & (branch_signs == int(branch_sign))
        ).nonzero(as_tuple=False).flatten()
        if not len(branch_rows):
          raise ValueError("v122 requires both mirrored branches for every pair")
        weights[branch_rows] = 1.0 / float(
          len(unique_seeds) * len(seed_pairs) * 2 * len(branch_rows)
        )
        advantages[branch_rows] = float(pair_advantage) * float(branch_sign)
  return weights, advantages


def select_joint_surrogate_scale(
  records: list[dict[str, Any]],
  *,
  training_before: float,
  validation_before: float,
  max_reference_kl: float,
  minimum_gain: float = MINIMUM_JOINT_SURROGATE_GAIN,
) -> float | None:
  """Choose the largest predefined scale improving both clipped objectives."""
  selected: list[float] = []
  for record in records:
    scale = float(record["scale"])
    training = record["training"]
    validation = record["validation"]
    values = (
      scale,
      float(training["clipped_surrogate"]),
      float(validation["clipped_surrogate"]),
      float(training["reference_forward_kl"]),
      float(validation["reference_forward_kl"]),
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("v123 scale-search metric is non-finite")
    if (
      0.0 < scale <= 1.0
      and values[1] >= float(training_before) + float(minimum_gain)
      and values[2] >= float(validation_before) + float(minimum_gain)
      and values[3] <= float(max_reference_kl)
      and values[4] <= float(max_reference_kl)
    ):
      selected.append(scale)
  return max(selected) if selected else None


def last_seed_transition_masks(
  environment_ids: torch.Tensor,
  *,
  num_envs: int,
  num_seeds: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Split complete rollout seeds without leaking validation transitions."""
  if (
    environment_ids.ndim != 1
    or environment_ids.dtype != torch.long
    or num_envs < 1
    or num_seeds < 2
  ):
    raise ValueError("v119 transition ids or seed dimensions are invalid")
  cutoff = int((num_seeds - 1) * num_envs)
  train = environment_ids < cutoff
  validation = environment_ids >= cutoff
  if not bool(train.any()) or not bool(validation.any()) or bool((train & validation).any()):
    raise ValueError("v119 train/validation seed split is empty or overlapping")
  return train, validation


def _subset_dataset(
  dataset: dict[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
  return {key: value[mask] for key, value in dataset.items()}


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
  parser.add_argument("--exploration-std", type=float, default=0.02)
  parser.add_argument("--max-residual", type=float, default=0.10)
  parser.add_argument("--learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
  parser.add_argument(
    "--validation-last-seed",
    action="store_true",
    help="Reserve the final complete rollout seed for transactional acceptance.",
  )
  parser.add_argument(
    "--episode-credit",
    choices=(
      "binary-balanced",
      "terminal-progress-standardized",
      "pretrained-critic-gae",
      "paired-antithetic-progress",
      "paired-antithetic-calibrated",
    ),
    default="binary-balanced",
  )
  parser.add_argument("--success-bonus", type=float, default=1.0)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=8192)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument("--moving-kl-beta", type=float, default=0.1)
  parser.add_argument("--max-reference-kl", type=float, default=0.02)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v117 seeds must be comma-separated integers") from exc
  if len(seeds) < 3 or len(set(seeds)) != len(seeds):
    raise ValueError("v117 requires at least three unique rollout seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v117 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _set_filter_off(action_term, num_envs: int, device) -> None:
  action_term.set_runtime_filter_mask(
    torch.zeros(num_envs, dtype=torch.bool, device=device)
  )


def _normal_log_prob(
  actions: torch.Tensor,
  means: torch.Tensor,
  std: float,
) -> torch.Tensor:
  variance = float(std) ** 2
  constant = math.log(2.0 * math.pi * variance)
  return -0.5 * (((actions - means) ** 2) / variance + constant).sum(dim=-1)


def _collect_rollout(
  runner,
  base_env,
  action_term,
  residual: LearnedCbfResidual,
  *,
  seed: int,
  exploration_std: float,
  max_residual: float,
  environment_offset: int,
  collect_critic_gae: bool = False,
  exploration_sign: float = 1.0,
) -> dict[str, Any]:
  if exploration_sign not in (-1.0, 1.0):
    raise ValueError("v122 exploration sign must be -1 or +1")
  _set_filter_off(action_term, base_env.num_envs, base_env.device)
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
  exploration_generator = torch.Generator(device=base_env.device)
  exploration_generator.manual_seed(int(seed))
  signature = _initial_state_signature(
    observations,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  success = torch.zeros_like(active)
  fell = torch.zeros_like(active)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  feature_chunks: list[torch.Tensor] = []
  sampled_chunks: list[torch.Tensor] = []
  mean_chunks: list[torch.Tensor] = []
  log_prob_chunks: list[torch.Tensor] = []
  environment_id_chunks: list[torch.Tensor] = []
  full_transition_index_chunks: list[torch.Tensor] = []
  critic_value_chunks: list[torch.Tensor] = []
  reward_chunks: list[torch.Tensor] = []
  done_chunks: list[torch.Tensor] = []
  critic_environment_id_chunks: list[torch.Tensor] = []
  episode_step_chunks: list[torch.Tensor] = []
  full_transition_count = 0
  active_geometry_transition_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  actor = runner.alg.actor
  actor.eval()
  residual.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      base_action = actor(observations, stochastic_output=False)
      critic_values = (
        runner.alg.critic(observations).reshape(-1)
        if collect_critic_gae
        else None
      )
      features = _features_from_observations(actor, observations)
      means = residual(features)
      exploration_noise = torch.randn(
        means.shape,
        generator=exploration_generator,
        device=means.device,
        dtype=means.dtype,
      )
      sampled = means + float(exploration_sign) * float(
        exploration_std
      ) * exploration_noise
      executed_residual = torch.clamp(
        sampled, -float(max_residual), float(max_residual)
      )
      geometry_active = _geometry_active_from_observations(observations)
      executed_residual = executed_residual * geometry_active.unsqueeze(-1)
      actions = base_action + executed_residual
      next_observations, rewards, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        ids_cpu = ids.cpu()
        current_full_indices = None
        if collect_critic_gae:
          assert critic_values is not None
          current_full_indices = torch.arange(
            full_transition_count,
            full_transition_count + len(ids),
            dtype=torch.long,
          )
          critic_value_chunks.append(critic_values[ids].float().cpu())
          reward_chunks.append(rewards.reshape(-1)[ids].float().cpu())
          done_chunks.append(dones.reshape(-1)[ids].bool().cpu())
          critic_environment_id_chunks.append(ids_cpu + int(environment_offset))
          episode_step_chunks.append(steps[ids].cpu())
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        selected_positions = geometry_active[ids].nonzero(
          as_tuple=False
        ).flatten()
        selected = ids[selected_positions]
        if selected.numel():
          feature_chunks.append(features[selected].cpu())
          sampled_chunks.append(sampled[selected].cpu())
          mean_chunks.append(means[selected].cpu())
          log_prob_chunks.append(
            _normal_log_prob(
              sampled[selected], means[selected], exploration_std
            ).cpu()
          )
          environment_id_chunks.append(
            selected.cpu() + int(environment_offset)
          )
          if collect_critic_gae:
            assert current_full_indices is not None
            full_transition_index_chunks.append(
              current_full_indices[selected_positions.cpu()]
            )
          active_geometry_transition_count += len(selected)
        if collect_critic_gae:
          full_transition_count += len(ids)
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
    raise RuntimeError("v117 did not finish every explored first episode")
  if not feature_chunks:
    raise RuntimeError("v117 collected no active-geometry residual transitions")
  result = {
    "seed": seed,
    "initial_state_signature": signature,
    "success": success.cpu(),
    "fell": fell.cpu(),
    "steps": steps.cpu(),
    "reached_risers": reached_risers.cpu(),
    "success_count": int(success.sum()),
    "fall_count": int(fell.sum()),
    "mean_reached_riser": float(reached_risers.float().mean()),
    "active_geometry_transition_count": active_geometry_transition_count,
    "exploration_sign": exploration_sign,
    "features": torch.cat(feature_chunks),
    "sampled_residuals": torch.cat(sampled_chunks),
    "old_means": torch.cat(mean_chunks),
    "old_log_prob": torch.cat(log_prob_chunks),
    "environment_ids": torch.cat(environment_id_chunks),
  }
  if collect_critic_gae:
    result["full_transition_indices"] = torch.cat(
      full_transition_index_chunks
    )
    result["critic_dataset"] = {
      "values": torch.cat(critic_value_chunks),
      "rewards": torch.cat(reward_chunks),
      "dones": torch.cat(done_chunks),
      "environment_ids": torch.cat(critic_environment_id_chunks),
      "episode_steps": torch.cat(episode_step_chunks),
    }
  return result


def _ppo_metrics(
  residual: LearnedCbfResidual,
  dataset: dict[str, torch.Tensor],
  weights: torch.Tensor,
  advantages: torch.Tensor,
  *,
  exploration_std: float,
  clip_ratio: float,
  device: str,
  batch_size: int,
) -> dict[str, float | int]:
  objective_sum = unclipped_sum = kl_sum = ratio_sum = weight_sum = 0.0
  count = 0
  residual.eval()
  with torch.inference_mode():
    for start in range(0, len(weights), batch_size):
      stop = min(start + batch_size, len(weights))
      features = dataset["features"][start:stop].to(device)
      sampled = dataset["sampled_residuals"][start:stop].to(device)
      old_means = dataset["old_means"][start:stop].to(device)
      old_log_prob = dataset["old_log_prob"][start:stop].to(device)
      effective = weights[start:stop].to(device)
      advantage = advantages[start:stop].to(device)
      means = residual(features)
      log_prob = _normal_log_prob(sampled, means, exploration_std)
      ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -10.0, 10.0))
      unclipped = ratio * advantage
      clipped = torch.clamp(
        ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)
      ) * advantage
      objective = torch.minimum(unclipped, clipped)
      kl = ((means - old_means) ** 2).sum(dim=-1) / (
        2.0 * float(exploration_std) ** 2
      )
      objective_sum += float((effective * objective).sum())
      unclipped_sum += float((effective * unclipped).sum())
      kl_sum += float((effective * kl).sum())
      ratio_sum += float((effective * ratio).sum())
      weight_sum += float(effective.sum())
      count += len(features)
  return {
    "transition_count": count,
    "weight_sum": weight_sum,
    "clipped_surrogate": objective_sum / max(1.0e-8, weight_sum),
    "unclipped_surrogate": unclipped_sum / max(1.0e-8, weight_sum),
    "reference_forward_kl": kl_sum / max(1.0e-8, weight_sum),
    "importance_ratio_mean": ratio_sum / max(1.0e-8, weight_sum),
  }


def _load_scaled_state(
  residual: LearnedCbfResidual,
  reference: dict[str, torch.Tensor],
  proposed: dict[str, torch.Tensor],
  scale: float,
) -> None:
  state = {
    key: reference[key] + float(scale) * (proposed[key] - reference[key])
    for key in reference
  }
  residual.load_state_dict(state, strict=True)


def _fit_residual_ppo(
  residual: LearnedCbfResidual,
  dataset: dict[str, torch.Tensor],
  weights: torch.Tensor,
  advantages: torch.Tensor,
  *,
  exploration_std: float,
  optimizer_name: str,
  learning_rate: float,
  epochs: int,
  batch_size: int,
  clip_ratio: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  reference = {
    key: value.detach().clone() for key, value in residual.state_dict().items()
  }
  before = _ppo_metrics(
    residual,
    dataset,
    weights,
    advantages,
    exploration_std=exploration_std,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  optimizer = _build_optimizer(
    residual, name=optimizer_name, learning_rate=learning_rate
  )
  updates = 0
  maximum_gradient_norm = 0.0
  residual.train()
  for _ in range(epochs):
    permutation = torch.randperm(len(weights))
    for start in range(0, len(permutation), batch_size):
      indices = permutation[start : start + batch_size]
      features = dataset["features"][indices].to(device)
      sampled = dataset["sampled_residuals"][indices].to(device)
      old_means = dataset["old_means"][indices].to(device)
      old_log_prob = dataset["old_log_prob"][indices].to(device)
      effective = weights[indices].to(device)
      advantage = advantages[indices].to(device)
      means = residual(features)
      log_prob = _normal_log_prob(sampled, means, exploration_std)
      ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -10.0, 10.0))
      unclipped = ratio * advantage
      clipped = torch.clamp(
        ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)
      ) * advantage
      surrogate = torch.minimum(unclipped, clipped)
      policy_loss = -(effective * surrogate).sum() / effective.sum().clamp_min(
        1.0e-8
      )
      moving_kl = ((means - old_means) ** 2).sum(dim=-1) / (
        2.0 * float(exploration_std) ** 2
      )
      loss = policy_loss + float(moving_kl_beta) * (
        (effective * moving_kl).sum() / effective.sum().clamp_min(1.0e-8)
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        residual.parameters(), max_grad_norm
      )
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v117 residual PPO gradient is non-finite")
      optimizer.step()
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      updates += 1
  unprojected = _ppo_metrics(
    residual,
    dataset,
    weights,
    advantages,
    exploration_std=exploration_std,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  proposed = {
    key: value.detach().clone() for key, value in residual.state_dict().items()
  }
  scale = 1.0
  projection_iterations = 0
  if unprojected["reference_forward_kl"] > max_reference_kl:
    low, high = 0.0, 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      _load_scaled_state(residual, reference, proposed, middle)
      metrics = _ppo_metrics(
        residual,
        dataset,
        weights,
        advantages,
        exploration_std=exploration_std,
        clip_ratio=clip_ratio,
        device=device,
        batch_size=batch_size,
      )
      if metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    scale = low
    _load_scaled_state(residual, reference, proposed, scale)
  after = _ppo_metrics(
    residual,
    dataset,
    weights,
    advantages,
    exploration_std=exploration_std,
    clip_ratio=clip_ratio,
    device=device,
    batch_size=batch_size,
  )
  return {
    "before": before,
    "unprojected_after": unprojected,
    "after": after,
    "epochs": epochs,
    "optimizer_updates": updates,
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
    "trust_region": {
      "max_reference_kl": max_reference_kl,
      "parameter_interpolation_scale": scale,
      "projection_iterations": projection_iterations,
    },
  }, optimizer


def _evaluate_filter_off(
  runner,
  base_env,
  action_term,
  residual: LearnedCbfResidual,
  *,
  seed: int,
  screen_envs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  _set_filter_off(action_term, base_env.num_envs, base_env.device)
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
  residual_norm_sum = 0.0
  residual_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  actor = runner.alg.actor
  actor.eval()
  residual.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      base_action = actor(observations, stochastic_output=False)
      features = _features_from_observations(actor, observations)
      geometry_active = _geometry_active_from_observations(observations)
      correction = residual(features) * geometry_active.unsqueeze(-1)
      next_observations, _, dones, extras = runner.env.step(
        base_action + correction
      )
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        residual_norm_sum += float(
          torch.linalg.vector_norm(correction[ids], dim=-1).sum()
        )
        residual_count += len(ids)
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
    raise RuntimeError("v117 did not finish the filter-off screen")
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
    "success_count": success_count,
    "success_rate": success_count / screen_envs,
    "fall_count": int(fell[considered].sum()),
    "fall_rate": float(fell[considered].float().mean()),
    "mean_reached_riser": float(reached_risers[considered].float().mean()),
    "mean_residual_norm": residual_norm_sum / max(1, residual_count),
    "all_finite": math.isfinite(residual_norm_sum),
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
  use_critic_gae = args.episode_credit == "pretrained-critic-gae"
  use_joint_scale_search = args.episode_credit == "paired-antithetic-calibrated"
  use_antithetic = args.episode_credit in (
    "paired-antithetic-progress",
    "paired-antithetic-calibrated",
  )
  if (use_critic_gae or use_antithetic) and not args.validation_last_seed:
    raise ValueError("v121/v122 requires a complete held-out rollout seed")
  if args.validation_last_seed:
    if args.optimizer != "sgd" or len(seeds) < 4:
      raise ValueError("v119 held-out validation requires SGD and four seeds")
    if use_critic_gae:
      method_id = CRITIC_GAE_VALIDATED_METHOD_ID
    elif use_joint_scale_search:
      method_id = CALIBRATED_ANTITHETIC_METHOD_ID
    elif use_antithetic:
      method_id = ANTITHETIC_VALIDATED_METHOD_ID
    elif args.episode_credit == "terminal-progress-standardized":
      method_id = PROGRESS_VALIDATED_METHOD_ID
    else:
      method_id = VALIDATED_FULL_BATCH_METHOD_ID
  else:
    method_id = FULL_BATCH_METHOD_ID if args.optimizer == "sgd" else METHOD_ID
  if args.num_envs < 2 or not 1 <= args.screen_envs <= args.num_envs:
    raise ValueError("v117 environment counts are invalid")
  if args.epochs < 1 or args.batch_size < 1:
    raise ValueError("v117 optimization dimensions must be positive")
  if not 0.005 <= args.exploration_std <= 0.1:
    raise ValueError("v117 exploration std must lie in [0.005, 0.1]")
  if not 0.01 <= args.max_residual <= 0.25:
    raise ValueError("v117 residual bound must lie in [0.01, 0.25]")
  if not 1.0e-5 <= args.learning_rate <= 1.0e-2:
    raise ValueError("v117 learning rate is outside the supported range")
  if not 0.0 < args.clip_ratio <= 0.4:
    raise ValueError("v117 clip ratio must lie in (0, 0.4]")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v117 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.1:
    raise ValueError("v117 reference KL cap must lie in (0, 0.1]")
  if not 0.0 <= args.success_bonus <= 4.0:
    raise ValueError("v120 success bonus must lie in [0, 4]")
  if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
    raise ValueError("v121 gamma/lambda are outside the supported range")

  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v117 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v117 checkpoint or search protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v117 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v117 base checkpoint SHA-256 differs")
  output.mkdir(parents=True)
  started = time.monotonic()
  source_commit = _git(repo, "rev-parse", "HEAD")
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": method_id,
      "git_commit": source_commit,
      "base_checkpoint_sha256": checkpoint_sha,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "screen_seed": args.screen_seed,
      "training_runtime_filter": False,
      "validation_last_seed": args.validation_last_seed,
      "episode_credit": args.episode_credit,
      "success_bonus": args.success_bonus,
      "gamma": args.gamma,
      "gae_lambda": args.gae_lambda,
      "paired_antithetic": use_antithetic,
      "joint_scale_search": use_joint_scale_search,
      "joint_scale_grid": JOINT_SCALE_GRID if use_joint_scale_search else None,
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
    env_cfg, "raw_moderate", runtime_filter_during_training=False
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
    raise RuntimeError("v117 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v117 requires the current velocity-CBF action")

  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    for parameter in runner.alg.actor.parameters():
      parameter.requires_grad_(False)
    expanded_critic = None
    critic_expansion = None
    base_critic_sha = None
    if use_critic_gae:
      expanded_critic, critic_expansion = _expand_critic_state(
        source_payload["critic_state_dict"], runner.alg.critic.state_dict()
      )
      runner.alg.critic.load_state_dict(expanded_critic, strict=True)
      runner.alg.critic.eval()
      for parameter in runner.alg.critic.parameters():
        parameter.requires_grad_(False)
      base_critic_sha = actor_state_sha256(expanded_critic)
    residual = LearnedCbfResidual(args.max_residual).to(args.device)
    initial_residual_sha = actor_state_sha256(actor_state(residual))

    dataset_chunks: dict[str, list[torch.Tensor]] = {
      "features": [],
      "sampled_residuals": [],
      "old_means": [],
      "old_log_prob": [],
      "environment_ids": [],
    }
    if use_critic_gae:
      dataset_chunks["full_transition_indices"] = []
    if use_antithetic:
      dataset_chunks["pair_ids"] = []
      dataset_chunks["branch_signs"] = []
    critic_dataset_chunks: dict[str, list[torch.Tensor]] = {
      "values": [],
      "rewards": [],
      "dones": [],
      "environment_ids": [],
      "episode_steps": [],
    }
    all_success: list[torch.Tensor] = []
    all_reached_risers: list[torch.Tensor] = []
    rollout_summaries: list[dict[str, Any]] = []
    critic_transition_offset = 0
    paired_score_branches = torch.zeros(
      (len(seeds) * args.num_envs, 2), dtype=torch.float32
    )
    paired_success_branches = torch.zeros(
      (len(seeds) * args.num_envs, 2), dtype=torch.bool
    )

    def record_rollout(
      rollout: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
      nonlocal critic_transition_offset
      if use_critic_gae:
        local_critic_dataset = rollout.pop("critic_dataset")
        rollout["full_transition_indices"] += critic_transition_offset
        for key in critic_dataset_chunks:
          critic_dataset_chunks[key].append(local_critic_dataset[key])
        critic_transition_offset += len(local_critic_dataset["values"])
      for key in dataset_chunks:
        dataset_chunks[key].append(rollout.pop(key))
      success = rollout.pop("success")
      rollout.pop("fell")
      rollout.pop("steps")
      reached_risers = rollout.pop("reached_risers")
      all_success.append(success)
      all_reached_risers.append(reached_risers)
      rollout_summaries.append(rollout)
      print(json.dumps({"outcome_rollout_completed": rollout}), flush=True)
      return success, reached_risers

    if use_antithetic:
      for seed_index, seed in enumerate(seeds):
        signatures: list[str] = []
        pair_offset = seed_index * args.num_envs
        for branch_index, exploration_sign in enumerate((1.0, -1.0)):
          environment_offset = (
            seed_index * 2 + branch_index
          ) * args.num_envs
          rollout = _collect_rollout(
            runner,
            base_env,
            action_term,
            residual,
            seed=seed,
            exploration_std=args.exploration_std,
            max_residual=args.max_residual,
            environment_offset=environment_offset,
            exploration_sign=exploration_sign,
          )
          signatures.append(str(rollout["initial_state_signature"]))
          rollout["pair_ids"] = (
            rollout["environment_ids"] - environment_offset + pair_offset
          )
          rollout["branch_signs"] = torch.full(
            (len(rollout["environment_ids"]),),
            int(exploration_sign),
            dtype=torch.int8,
          )
          success, reached_risers = record_rollout(rollout)
          pair_rows = slice(pair_offset, pair_offset + args.num_envs)
          paired_success_branches[pair_rows, branch_index] = success
          paired_score_branches[pair_rows, branch_index] = (
            reached_risers.float() + float(args.success_bonus) * success.float()
          )
        if signatures[0] != signatures[1]:
          raise RuntimeError("v122 mirrored initial-state signatures differ")
    else:
      environment_offset = 0
      for seed in seeds:
        rollout = _collect_rollout(
          runner,
          base_env,
          action_term,
          residual,
          seed=seed,
          exploration_std=args.exploration_std,
          max_residual=args.max_residual,
          environment_offset=environment_offset,
          collect_critic_gae=use_critic_gae,
        )
        record_rollout(rollout)
        environment_offset += args.num_envs

    dataset = {key: torch.cat(chunks) for key, chunks in dataset_chunks.items()}
    critic_dataset = (
      {
        key: torch.cat(chunks)
        for key, chunks in critic_dataset_chunks.items()
      }
      if use_critic_gae
      else None
    )
    full_advantages = (
      compute_pretrained_critic_gae(
        critic_dataset,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
      )
      if critic_dataset is not None
      else None
    )
    gae_summary = (
      {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "full_transition_count": len(full_advantages),
        "active_geometry_transition_count": len(dataset["environment_ids"]),
        "advantage_mean": float(full_advantages.mean()),
        "advantage_std": float(full_advantages.std(unbiased=False)),
        "advantage_min": float(full_advantages.min()),
        "advantage_max": float(full_advantages.max()),
        "all_finite": bool(torch.isfinite(full_advantages).all()),
      }
      if full_advantages is not None
      else None
    )
    pair_score_differences = (
      paired_score_branches[:, 0] - paired_score_branches[:, 1]
      if use_antithetic
      else None
    )
    antithetic_summary = (
      {
        "pair_count": len(pair_score_differences),
        "plus_success_count": int(paired_success_branches[:, 0].sum()),
        "minus_success_count": int(paired_success_branches[:, 1].sum()),
        "positive_difference_count": int((pair_score_differences > 0).sum()),
        "negative_difference_count": int((pair_score_differences < 0).sum()),
        "zero_difference_count": int((pair_score_differences == 0).sum()),
        "score_difference_mean": float(pair_score_differences.mean()),
        "score_difference_std": float(
          pair_score_differences.std(unbiased=False)
        ),
        "matched_initial_state_signatures": True,
      }
      if pair_score_differences is not None
      else None
    )
    episode_success = torch.cat(all_success)
    episode_reached_risers = torch.cat(all_reached_risers)
    validation_dataset = None
    validation_weights = validation_advantages = None
    if args.validation_last_seed:
      train_mask, validation_mask = last_seed_transition_masks(
        dataset["environment_ids"],
        num_envs=args.num_envs * (2 if use_antithetic else 1),
        num_seeds=len(seeds),
      )
      training_dataset = _subset_dataset(dataset, train_mask)
      validation_dataset = _subset_dataset(dataset, validation_mask)
      if use_antithetic:
        assert pair_score_differences is not None
        weights, advantages = standardized_antithetic_weights(
          training_dataset["pair_ids"],
          training_dataset["branch_signs"],
          pair_score_differences,
          num_pairs_per_seed=args.num_envs,
        )
        validation_weights, validation_advantages = (
          standardized_antithetic_weights(
            validation_dataset["pair_ids"],
            validation_dataset["branch_signs"],
            pair_score_differences,
            num_pairs_per_seed=args.num_envs,
          )
        )
      elif use_critic_gae:
        assert full_advantages is not None
        weights, advantages = standardized_gae_weights(
          training_dataset["environment_ids"],
          training_dataset["full_transition_indices"],
          full_advantages,
          num_envs=args.num_envs,
        )
        validation_weights, validation_advantages = standardized_gae_weights(
          validation_dataset["environment_ids"],
          validation_dataset["full_transition_indices"],
          full_advantages,
          num_envs=args.num_envs,
        )
      elif args.episode_credit == "terminal-progress-standardized":
        weights, advantages = standardized_progress_weights(
          training_dataset["environment_ids"],
          episode_success,
          episode_reached_risers,
          num_envs=args.num_envs,
          success_bonus=args.success_bonus,
        )
        validation_weights, validation_advantages = standardized_progress_weights(
          validation_dataset["environment_ids"],
          episode_success,
          episode_reached_risers,
          num_envs=args.num_envs,
          success_bonus=args.success_bonus,
        )
      else:
        weights, advantages = balanced_outcome_weights(
          training_dataset["environment_ids"], episode_success
        )
        validation_weights, validation_advantages = balanced_outcome_weights(
          validation_dataset["environment_ids"], episode_success
        )
    else:
      training_dataset = dataset
      if use_antithetic:
        assert pair_score_differences is not None
        weights, advantages = standardized_antithetic_weights(
          training_dataset["pair_ids"],
          training_dataset["branch_signs"],
          pair_score_differences,
          num_pairs_per_seed=args.num_envs,
        )
      elif use_critic_gae:
        assert full_advantages is not None
        weights, advantages = standardized_gae_weights(
          training_dataset["environment_ids"],
          training_dataset["full_transition_indices"],
          full_advantages,
          num_envs=args.num_envs,
        )
      elif args.episode_credit == "terminal-progress-standardized":
        weights, advantages = standardized_progress_weights(
          training_dataset["environment_ids"],
          episode_success,
          episode_reached_risers,
          num_envs=args.num_envs,
          success_bonus=args.success_bonus,
        )
      else:
        weights, advantages = balanced_outcome_weights(
          training_dataset["environment_ids"], episode_success
        )
    if args.optimizer == "sgd" and (
      args.epochs != 1 or args.batch_size < len(weights)
    ):
      raise ValueError(
        "v118 SGD requires exactly one epoch and one full transition batch"
      )
    _seed_everything(args.optimization_seed)
    reference_residual_state = {
      key: value.detach().clone() for key, value in residual.state_dict().items()
    }
    validation_before = None
    if validation_dataset is not None:
      assert validation_weights is not None and validation_advantages is not None
      validation_before = _ppo_metrics(
        residual,
        validation_dataset,
        validation_weights,
        validation_advantages,
        exploration_std=args.exploration_std,
        clip_ratio=args.clip_ratio,
        device=args.device,
        batch_size=args.batch_size,
      )
    training, optimizer = _fit_residual_ppo(
      residual,
      training_dataset,
      weights,
      advantages,
      exploration_std=args.exploration_std,
      optimizer_name=args.optimizer,
      learning_rate=args.learning_rate,
      epochs=args.epochs,
      batch_size=args.batch_size,
      clip_ratio=args.clip_ratio,
      moving_kl_beta=args.moving_kl_beta,
      max_reference_kl=args.max_reference_kl,
      max_grad_norm=args.max_grad_norm,
      device=args.device,
    )
    validation_after = None
    joint_scale_search = None
    if use_joint_scale_search:
      assert validation_dataset is not None
      assert validation_before is not None
      assert validation_weights is not None and validation_advantages is not None
      projected_residual_state = actor_state(residual)
      projected_training_after = copy.deepcopy(training["after"])
      scale_records: list[dict[str, Any]] = []
      for scale in JOINT_SCALE_GRID:
        _load_scaled_state(
          residual,
          reference_residual_state,
          projected_residual_state,
          scale,
        )
        scale_records.append(
          {
            "scale": scale,
            "training": _ppo_metrics(
              residual,
              training_dataset,
              weights,
              advantages,
              exploration_std=args.exploration_std,
              clip_ratio=args.clip_ratio,
              device=args.device,
              batch_size=args.batch_size,
            ),
            "validation": _ppo_metrics(
              residual,
              validation_dataset,
              validation_weights,
              validation_advantages,
              exploration_std=args.exploration_std,
              clip_ratio=args.clip_ratio,
              device=args.device,
              batch_size=args.batch_size,
            ),
          }
        )
      selected_scale = select_joint_surrogate_scale(
        scale_records,
        training_before=float(training["before"]["clipped_surrogate"]),
        validation_before=float(validation_before["clipped_surrogate"]),
        max_reference_kl=args.max_reference_kl,
      )
      if selected_scale is None:
        _load_scaled_state(
          residual,
          reference_residual_state,
          projected_residual_state,
          0.0,
        )
        training["after"] = copy.deepcopy(training["before"])
        validation_after = copy.deepcopy(validation_before)
      else:
        _load_scaled_state(
          residual,
          reference_residual_state,
          projected_residual_state,
          selected_scale,
        )
        selected_record = next(
          record
          for record in scale_records
          if float(record["scale"]) == float(selected_scale)
        )
        training["after"] = copy.deepcopy(selected_record["training"])
        validation_after = copy.deepcopy(selected_record["validation"])
      training["train_kl_projected_after"] = projected_training_after
      joint_scale_search = {
        "grid": JOINT_SCALE_GRID,
        "minimum_joint_surrogate_gain": MINIMUM_JOINT_SURROGATE_GAIN,
        "selected_scale_relative_to_train_kl_proposal": selected_scale,
        "effective_scale_relative_to_unprojected_proposal": (
          float(training["trust_region"]["parameter_interpolation_scale"])
          * float(selected_scale)
          if selected_scale is not None
          else 0.0
        ),
        "records": scale_records,
      }
      training["heldout_scale_search"] = joint_scale_search
    elif validation_dataset is not None:
      assert validation_weights is not None and validation_advantages is not None
      validation_after = _ppo_metrics(
        residual,
        validation_dataset,
        validation_weights,
        validation_advantages,
        exploration_std=args.exploration_std,
        clip_ratio=args.clip_ratio,
        device=args.device,
        batch_size=args.batch_size,
      )
    final_residual_state = actor_state(residual)
    final_residual_sha = actor_state_sha256(final_residual_state)
    offline_gate_passed = bool(
      training["after"]["clipped_surrogate"]
      > training["before"]["clipped_surrogate"]
      and training["after"]["reference_forward_kl"] <= args.max_reference_kl
      and (gae_summary is None or gae_summary["all_finite"])
      and (
        validation_after is None
        or (
          validation_before is not None
          and validation_after["clipped_surrogate"]
          > validation_before["clipped_surrogate"]
        )
      )
    )
    rolled_back = bool(args.validation_last_seed and not offline_gate_passed)
    if rolled_back:
      residual.load_state_dict(reference_residual_state, strict=True)
      final_residual_state = actor_state(residual)
      final_residual_sha = actor_state_sha256(final_residual_state)
    candidate_path = output / "candidate.pt"
    _atomic_torch(
      candidate_path,
      {
        "schema_version": 1,
        "method_id": method_id,
        "git_commit": source_commit,
        "base_checkpoint_sha256": checkpoint_sha,
        "base_actor_state_dict": {
          key: value.detach().cpu() for key, value in expanded_state.items()
        },
        "base_actor_sha256": actor_state_sha256(expanded_state),
        "base_critic_sha256": base_critic_sha,
        "critic_expansion": critic_expansion,
        "residual_state_dict": {
          key: value.detach().cpu() for key, value in final_residual_state.items()
        },
        "residual_state_sha256": final_residual_sha,
        "residual_config": {
          "max_residual": args.max_residual,
          "exploration_std": args.exploration_std,
          "geometry_active_only": True,
          "hidden_dims": [128, 64],
        },
        "offline_gate_passed": offline_gate_passed,
        "episode_credit": args.episode_credit,
        "success_bonus": args.success_bonus,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "gae_summary": gae_summary,
        "antithetic_summary": antithetic_summary,
        "joint_scale_search": joint_scale_search,
        "heldout_validation": {
          "seed": seeds[-1],
          "before": validation_before,
          "after": validation_after,
        }
        if args.validation_last_seed
        else None,
        "transactional_rollback": rolled_back,
      },
    )
    screen, screen_rows = _evaluate_filter_off(
      runner,
      base_env,
      action_term,
      residual,
      seed=args.screen_seed,
      screen_envs=args.screen_envs,
    )
    screen["candidate_checkpoint_sha256"] = file_sha256(candidate_path)
    screen["residual_state_sha256"] = final_residual_sha
    _atomic_json(output / "screen_summary.json", screen)
    _write_csv(output / "screen_episodes.csv", screen_rows)

    summary = {
      "schema_version": 1,
      "method_id": method_id,
      "git_commit": source_commit,
      "context": args.context,
      "base_checkpoint_sha256": checkpoint_sha,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "base_actor_sha256": actor_state_sha256(expanded_state),
      "base_critic_sha256": base_critic_sha,
      "initial_residual_sha256": initial_residual_sha,
      "final_residual_sha256": final_residual_sha,
      "trainable_parameter_count": sum(
        parameter.numel() for parameter in residual.parameters()
      ),
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "optimization_seed": args.optimization_seed,
      "training_runtime_filter": False,
      "exploration_std": args.exploration_std,
      "max_residual": args.max_residual,
      "optimizer": args.optimizer,
      "validation_last_seed": args.validation_last_seed,
      "training_seed_count": len(seeds) - int(args.validation_last_seed),
      "validation_seed": seeds[-1] if args.validation_last_seed else None,
      "episode_credit": args.episode_credit,
      "success_bonus": args.success_bonus,
      "gamma": args.gamma,
      "gae_lambda": args.gae_lambda,
      "gae_summary": gae_summary,
      "antithetic_summary": antithetic_summary,
      "joint_scale_search": joint_scale_search,
      "training_advantage_mean": float(advantages.mean()),
      "training_advantage_std": float(advantages.std(unbiased=False)),
      "validation_advantage_mean": (
        float(validation_advantages.mean())
        if validation_advantages is not None
        else None
      ),
      "validation_advantage_std": (
        float(validation_advantages.std(unbiased=False))
        if validation_advantages is not None
        else None
      ),
      "episode_success_count": int(episode_success.sum()),
      "episode_failure_count": int((~episode_success).sum()),
      "training_transition_count": len(weights),
      "validation_transition_count": (
        len(validation_weights) if validation_weights is not None else 0
      ),
      "rollout_summaries": rollout_summaries,
      "training": training,
      "validation": {
        "before": validation_before,
        "after": validation_after,
      }
      if args.validation_last_seed
      else None,
      "offline_gate_passed": offline_gate_passed,
      "transactional_rollback": rolled_back,
      "screen": screen,
      "independent_gate_run": False,
      "independent_gate_policy": "run separately only if screen_rate_gte_0.75",
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "critic_expansion": critic_expansion,
      "optimizer_state_in_candidate": False,
      "optimizer_parameter_state_count": len(optimizer.state),
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
