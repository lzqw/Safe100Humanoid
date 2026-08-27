"""Internalize the training-time CBF with a deployable learned residual head.

The base task policy is frozen. During DAgger rounds it proposes a deterministic
nominal action, a small residual network adds a deployable correction, and the
closed-form CBF filters the combined action before simulator execution. The
residual network directly learns the filtered-minus-nominal direction behind
the paper's Eq. (23). At the final gate the runtime filter is disabled and only
the base policy plus learned residual network executes.
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
from refine_observable_cbf_adapter_v49 import _expand_actor_state
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "learned-cbf-residual-policy-dagger-v97"
BASE_HIDDEN_DIM = 128
PERSISTENT_GEOMETRY_DIM = 10
ACTION_DIM = 12
RESIDUAL_INPUT_DIM = BASE_HIDDEN_DIM + PERSISTENT_GEOMETRY_DIM + ACTION_DIM


class LearnedCbfResidual(torch.nn.Module):
  """Bounded deployable residual policy initialized as an exact no-op."""

  def __init__(self, max_residual: float, hidden_dims: tuple[int, int] = (128, 64)):
    super().__init__()
    self.max_residual = float(max_residual)
    self.network = torch.nn.Sequential(
      torch.nn.Linear(RESIDUAL_INPUT_DIM, hidden_dims[0]),
      torch.nn.ELU(),
      torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
      torch.nn.ELU(),
      torch.nn.Linear(hidden_dims[1], ACTION_DIM),
    )
    final = self.network[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    raw = self.network(features)
    return self.max_residual * torch.tanh(raw / self.max_residual)


def residual_teacher_target(
  current_residual: torch.Tensor,
  safe_minus_nominal: torch.Tensor,
  *,
  eta: float,
  max_residual: float,
) -> torch.Tensor:
  """Take a bounded DAgger step toward the filtered action."""
  return torch.clamp(
    current_residual + float(eta) * safe_minus_nominal,
    -float(max_residual),
    float(max_residual),
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
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--gate-envs", type=int, default=64)
  parser.add_argument("--gate-seed", type=int, required=True)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--teacher-eta", type=float, default=1.0)
  parser.add_argument("--max-residual", type=float, default=0.25)
  parser.add_argument("--learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--epochs-per-round", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--trust-weight", type=float, default=0.05)
  parser.add_argument("--max-grad-norm", type=float, default=5.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v97 seeds must be comma-separated integers") from exc
  if len(seeds) < 2 or len(set(seeds)) != len(seeds):
    raise ValueError("v97 requires at least two unique DAgger seeds")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v97 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _base_hidden(actor, observations) -> torch.Tensor:
  latent = actor.get_latent(observations)
  hidden = latent
  for layer in list(actor.mlp.children())[:-1]:
    hidden = layer(hidden)
  if hidden.shape[-1] != BASE_HIDDEN_DIM:
    raise RuntimeError(f"v97 expected {BASE_HIDDEN_DIM}-D base hidden state")
  return hidden


def _policy_step(actor, residual, observations):
  base_action = actor(observations, stochastic_output=False)
  geometry = observations["cbf_geometry"]
  if geometry.shape[-1] != PERSISTENT_GEOMETRY_DIM:
    raise RuntimeError("v97 requires 10-D persistent geometry")
  hidden = _base_hidden(actor, observations)
  features = torch.cat((hidden, geometry, base_action), dim=-1)
  correction = residual(features)
  return base_action + correction, features, correction


def _set_filter(action_term, enabled: bool, num_envs: int, device) -> None:
  action_term.set_runtime_filter_mask(
    torch.full((num_envs,), enabled, dtype=torch.bool, device=device)
  )


def _collect_dagger_round(
  runner,
  base_env,
  action_term,
  residual,
  *,
  seed: int,
  teacher_eta: float,
  max_residual: float,
) -> dict[str, Any]:
  _set_filter(action_term, True, base_env.num_envs, base_env.device)
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
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
  teacher_features: list[torch.Tensor] = []
  teacher_targets: list[torch.Tensor] = []
  teacher_corrections: list[torch.Tensor] = []
  trust_features: list[torch.Tensor] = []
  trust_targets: list[torch.Tensor] = []
  transition_count = intervention_count = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  actor = runner.alg.actor
  actor.eval()
  residual.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions, features, current_residual = _policy_step(
        actor, residual, observations
      )
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        transition_count += len(ids)
        trust_features.append(features[ids].cpu())
        trust_targets.append(current_residual[ids].cpu())
        intervened = extras["cbf_would_intervene"][ids].bool()
        if bool(intervened.any()):
          selected = ids[intervened]
          correction = (
            extras["cbf_safe_raw_action"][selected]
            - extras["cbf_nominal_raw_action"][selected]
          )
          target = residual_teacher_target(
            current_residual[selected],
            correction,
            eta=teacher_eta,
            max_residual=max_residual,
          )
          teacher_features.append(features[selected].cpu())
          teacher_targets.append(target.cpu())
          teacher_corrections.append(correction.cpu())
          intervention_count += len(selected)
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
    raise RuntimeError("v97 did not finish every DAgger first episode")
  if not teacher_features:
    raise RuntimeError("v97 collected no CBF interventions")
  corrections = torch.cat(teacher_corrections)
  return {
    "seed": seed,
    "initial_state_signature": signature,
    "success": success.cpu(),
    "fell": fell.cpu(),
    "steps": steps.cpu(),
    "success_count": int(success.sum()),
    "fall_count": int(fell.sum()),
    "transition_count": transition_count,
    "intervention_count": intervention_count,
    "intervention_fraction": intervention_count / transition_count,
    "correction_norm_mean": float(torch.linalg.vector_norm(corrections, dim=-1).mean()),
    "correction_norm_max": float(torch.linalg.vector_norm(corrections, dim=-1).max()),
    "teacher_features": torch.cat(teacher_features),
    "teacher_targets": torch.cat(teacher_targets),
    "trust_features": torch.cat(trust_features),
    "trust_targets": torch.cat(trust_targets),
  }


def _residual_metrics(
  residual,
  features: torch.Tensor,
  targets: torch.Tensor,
  *,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  distance_sum = loss_sum = prediction_norm_sum = 0.0
  count = 0
  residual.eval()
  with torch.inference_mode():
    for start in range(0, len(features), batch_size):
      batch = features[start : start + batch_size].to(device)
      target = targets[start : start + batch_size].to(device)
      prediction = residual(batch)
      distance_sum += float(torch.linalg.vector_norm(prediction - target, dim=-1).sum())
      prediction_norm_sum += float(torch.linalg.vector_norm(prediction, dim=-1).sum())
      loss_sum += float(
        F.smooth_l1_loss(
          prediction, target, reduction="none", beta=0.05
        ).mean(dim=-1).sum()
      )
      count += len(batch)
  return {
    "transition_count": count,
    "target_distance": distance_sum / max(1, count),
    "smooth_l1": loss_sum / max(1, count),
    "prediction_norm": prediction_norm_sum / max(1, count),
  }


def _fit_residual(
  residual,
  teacher_features: torch.Tensor,
  teacher_targets: torch.Tensor,
  trust_features: torch.Tensor,
  trust_targets: torch.Tensor,
  optimizer,
  *,
  epochs: int,
  batch_size: int,
  trust_weight: float,
  max_grad_norm: float,
  device: str,
) -> dict[str, Any]:
  before = _residual_metrics(
    residual,
    teacher_features,
    teacher_targets,
    device=device,
    batch_size=batch_size,
  )
  residual.train()
  updates = 0
  maximum_gradient_norm = 0.0
  teacher_loss_sum = trust_loss_sum = 0.0
  for _ in range(epochs):
    permutation = torch.randperm(len(teacher_features))
    for start in range(0, len(permutation), batch_size):
      indices = permutation[start : start + batch_size]
      features = teacher_features[indices].to(device)
      targets = teacher_targets[indices].to(device)
      prediction = residual(features)
      teacher_loss = F.smooth_l1_loss(prediction, targets, beta=0.05)
      trust_count = min(len(indices), len(trust_features))
      trust_indices = torch.randint(len(trust_features), (trust_count,))
      trust_prediction = residual(trust_features[trust_indices].to(device))
      trust_target = trust_targets[trust_indices].to(device)
      trust_loss = F.smooth_l1_loss(
        trust_prediction, trust_target, beta=0.05
      )
      loss = teacher_loss + trust_weight * trust_loss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        residual.parameters(), max_grad_norm
      )
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v97 residual gradient is non-finite")
      optimizer.step()
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      teacher_loss_sum += float(teacher_loss.detach())
      trust_loss_sum += float(trust_loss.detach())
      updates += 1
  after = _residual_metrics(
    residual,
    teacher_features,
    teacher_targets,
    device=device,
    batch_size=batch_size,
  )
  return {
    "before": before,
    "after": after,
    "epochs": epochs,
    "optimizer_updates": updates,
    "mean_teacher_loss_during_update": teacher_loss_sum / max(1, updates),
    "mean_trust_loss_during_update": trust_loss_sum / max(1, updates),
    "maximum_gradient_norm_pre_clip": maximum_gradient_norm,
  }


def _evaluate_filter_off(
  runner,
  base_env,
  action_term,
  residual,
  *,
  seed: int,
  gate_envs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  _set_filter(action_term, False, base_env.num_envs, base_env.device)
  _seed_everything(seed)
  base_env.seed(seed)
  observations, _ = runner.env.reset()
  signature = _initial_state_signature(
    observations,
    base_env,
    action_term,
    base_env.command_manager.get_term("twist"),
  )
  considered = torch.arange(base_env.num_envs, device=base_env.device) < gate_envs
  active = considered.clone()
  success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  fell = torch.zeros_like(success)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  residual_norm_sum = 0.0
  residual_count = 0
  counterfactual_interventions = 0
  maximum_steps = int(base_env.max_episode_length) + 2
  actor = runner.alg.actor
  actor.eval()
  residual.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions, _, correction = _policy_step(actor, residual, observations)
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        residual_norm_sum += float(torch.linalg.vector_norm(correction[ids], dim=-1).sum())
        residual_count += len(ids)
        counterfactual_interventions += int(
          extras["cbf_would_intervene"][ids].sum()
        )
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
    raise RuntimeError("v97 did not finish the final filter-off gate")
  selected = considered.nonzero(as_tuple=False).flatten().cpu()
  rows = [
    {
      "environment_id": int(env_id),
      "success": bool(success[env_id]),
      "fell": bool(fell[env_id]),
      "steps": int(steps[env_id]),
      "reached_risers": int(reached_risers[env_id]),
    }
    for env_id in selected
  ]
  success_count = int(success[considered].sum())
  return {
    "seed": seed,
    "num_episodes": gate_envs,
    "initial_state_signature": signature,
    "runtime_filter": False,
    "success_count": success_count,
    "success_rate": success_count / gate_envs,
    "fall_count": int(fell[considered].sum()),
    "fall_rate": float(fell[considered].float().mean()),
    "mean_reached_riser": float(reached_risers[considered].float().mean()),
    "mean_residual_norm": residual_norm_sum / max(1, residual_count),
    "counterfactual_intervention_fraction": (
      counterfactual_interventions / max(1, residual_count)
    ),
    "all_finite": math.isfinite(residual_norm_sum),
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
  if args.num_envs < 2 or not 1 <= args.gate_envs <= args.num_envs:
    raise ValueError("v97 environment counts are invalid")
  if args.epochs_per_round < 1 or args.batch_size < 1:
    raise ValueError("v97 optimization dimensions must be positive")
  if not 0.0 < args.teacher_eta <= 1.0:
    raise ValueError("v97 teacher eta must lie in (0, 1]")
  if not 0.01 <= args.max_residual <= 1.0:
    raise ValueError("v97 residual bound must lie in [0.01, 1]")
  if not 1.0e-5 <= args.learning_rate <= 1.0e-2:
    raise ValueError("v97 learning rate is outside the supported range")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v97 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v97 checkpoint or protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v97 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v97 base checkpoint SHA-256 differs")
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
      "gate_seed": args.gate_seed,
      "gate_envs": args.gate_envs,
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
    raise RuntimeError("v97 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v97 requires the current velocity-CBF action")
  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    for parameter in runner.alg.actor.parameters():
      parameter.requires_grad_(False)
    residual = LearnedCbfResidual(args.max_residual).to(args.device)
    initial_residual_hash = actor_state_sha256(actor_state(residual))
    optimizer = torch.optim.Adam(residual.parameters(), lr=args.learning_rate)
    trainable_parameter_count = sum(
      parameter.numel() for parameter in residual.parameters()
    )
    replay_teacher_features: list[torch.Tensor] = []
    replay_teacher_targets: list[torch.Tensor] = []
    replay_trust_features: list[torch.Tensor] = []
    replay_trust_targets: list[torch.Tensor] = []
    round_summaries: list[dict[str, Any]] = []
    _seed_everything(args.optimization_seed)
    for round_index, seed in enumerate(seeds, start=1):
      round_started = time.monotonic()
      rollout = _collect_dagger_round(
        runner,
        base_env,
        action_term,
        residual,
        seed=seed,
        teacher_eta=args.teacher_eta,
        max_residual=args.max_residual,
      )
      replay_teacher_features.append(rollout.pop("teacher_features"))
      replay_teacher_targets.append(rollout.pop("teacher_targets"))
      replay_trust_features.append(rollout.pop("trust_features"))
      replay_trust_targets.append(rollout.pop("trust_targets"))
      _seed_everything(args.optimization_seed + round_index)
      fit = _fit_residual(
        residual,
        torch.cat(replay_teacher_features),
        torch.cat(replay_teacher_targets),
        torch.cat(replay_trust_features),
        torch.cat(replay_trust_targets),
        optimizer,
        epochs=args.epochs_per_round,
        batch_size=args.batch_size,
        trust_weight=args.trust_weight,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
      )
      summary = {
        "round": round_index,
        **rollout,
        "replay_teacher_transition_count": sum(
          len(chunk) for chunk in replay_teacher_features
        ),
        "replay_trust_transition_count": sum(
          len(chunk) for chunk in replay_trust_features
        ),
        "fit": fit,
        "residual_state_sha256": actor_state_sha256(actor_state(residual)),
        "elapsed_seconds": time.monotonic() - round_started,
      }
      summary.pop("success")
      summary.pop("fell")
      summary.pop("steps")
      round_summaries.append(summary)
      _atomic_json(output / "round_summaries.json", round_summaries)
      print(json.dumps({"dagger_round_completed": summary}), flush=True)

    final_residual_state = actor_state(residual)
    final_residual_hash = actor_state_sha256(final_residual_state)
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
        "residual_state_dict": {
          key: value.detach().cpu() for key, value in final_residual_state.items()
        },
        "residual_state_sha256": final_residual_hash,
        "residual_config": {
          "input_dim": RESIDUAL_INPUT_DIM,
          "base_hidden_dim": BASE_HIDDEN_DIM,
          "geometry_dim": PERSISTENT_GEOMETRY_DIM,
          "action_dim": ACTION_DIM,
          "hidden_dims": [128, 64],
          "max_residual": args.max_residual,
        },
        "round_summaries": round_summaries,
      },
    )
    gate, gate_rows = _evaluate_filter_off(
      runner,
      base_env,
      action_term,
      residual,
      seed=args.gate_seed,
      gate_envs=args.gate_envs,
    )
    gate["candidate_checkpoint_sha256"] = file_sha256(candidate_path)
    gate["residual_state_sha256"] = final_residual_hash
    gate["passed_75_percent"] = gate["success_rate"] >= 0.75
    _atomic_json(output / "untouched_filter_off_gate.json", gate)
    _write_csv(output / "untouched_filter_off_gate.csv", gate_rows)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "context": args.context,
      "base_checkpoint_sha256": checkpoint_sha,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "base_actor_sha256": actor_state_sha256(expanded_state),
      "initial_residual_sha256": initial_residual_hash,
      "final_residual_sha256": final_residual_hash,
      "trainable_parameter_count": trainable_parameter_count,
      "actor_observation_dim": 415,
      "training_seeds": seeds,
      "num_envs": args.num_envs,
      "teacher_eta": args.teacher_eta,
      "max_residual": args.max_residual,
      "learning_rate": args.learning_rate,
      "epochs_per_round": args.epochs_per_round,
      "batch_size": args.batch_size,
      "trust_weight": args.trust_weight,
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "round_summaries": round_summaries,
      "untouched_filter_off_gate": gate,
      "selected": gate["passed_75_percent"],
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
