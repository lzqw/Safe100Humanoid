"""Directly optimize deterministic residual parameters with paired ES returns."""

from __future__ import annotations

import argparse
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
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from train_filter_off_residual_ppo_v117 import (
  _evaluate_filter_off,
  _set_filter_off,
  _write_csv,
)
from train_paired_gated_residual_v113 import (
  _features_from_observations,
  _geometry_active_from_observations,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "heldout-validated-parameter-antithetic-residual-es-v124"
LOCAL_METHOD_ID = "heldout-validated-local-parameter-antithetic-residual-es-v125"
BASE_HIDDEN_DIM = 128
GEOMETRY_DIM = 10
ACTION_DIM = 12
FEATURE_DIM = 150
PARAMETER_WIDTH = GEOMETRY_DIM + 1


def parameter_method_id(local_search: bool) -> str:
  return LOCAL_METHOD_ID if local_search else METHOD_ID


class LinearGeometryResidual(torch.nn.Module):
  """A bounded, exactly-zero-initialized 10-D geometry residual."""

  def __init__(self, max_residual: float):
    super().__init__()
    self.max_residual = float(max_residual)
    self.linear = torch.nn.Linear(GEOMETRY_DIM, ACTION_DIM)
    torch.nn.init.zeros_(self.linear.weight)
    torch.nn.init.zeros_(self.linear.bias)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
      raise ValueError("v124 residual feature width differs")
    geometry = features[:, BASE_HIDDEN_DIM : BASE_HIDDEN_DIM + GEOMETRY_DIM]
    raw = self.linear(geometry)
    return self.max_residual * torch.tanh(raw / self.max_residual)


def standardized_parameter_gradient(
  directions: torch.Tensor,
  score_differences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Estimate one deterministic-policy ES gradient from matched pairs."""
  if (
    directions.ndim != 3
    or directions.shape[1:] != (ACTION_DIM, PARAMETER_WIDTH)
    or score_differences.shape != directions.shape[:1]
    or not len(directions)
    or not bool(torch.isfinite(directions).all())
    or not bool(torch.isfinite(score_differences).all())
  ):
    raise ValueError("v124 parameter directions or scores are invalid")
  scores = score_differences.float()
  standardized = (scores - scores.mean()) / scores.std(
    unbiased=False
  ).clamp_min(1.0e-6)
  gradient = (standardized[:, None, None] * directions.float()).mean(dim=0)
  return gradient, standardized


def calibrated_parameter_delta(
  gradient: torch.Tensor,
  geometry_samples: torch.Tensor,
  *,
  target_mean_norm: float,
  max_residual: float,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Scale one parameter direction to a target deterministic correction norm."""
  if (
    gradient.shape != (ACTION_DIM, PARAMETER_WIDTH)
    or geometry_samples.ndim != 2
    or geometry_samples.shape[1] != GEOMETRY_DIM
    or not len(geometry_samples)
    or not 0.0 < target_mean_norm < max_residual
  ):
    raise ValueError("v124 parameter calibration inputs are invalid")
  gradient_norm = torch.linalg.vector_norm(gradient.float())
  if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 1.0e-10:
    raise ValueError("v124 parameter gradient is zero or non-finite")
  unit = gradient.float() / gradient_norm
  augmented = torch.cat(
    (geometry_samples.float(), torch.ones((len(geometry_samples), 1))), dim=-1
  )

  def mean_norm(scale: float) -> float:
    raw = float(scale) * torch.einsum("ni,ai->na", augmented, unit)
    bounded = float(max_residual) * torch.tanh(raw / float(max_residual))
    return float(torch.linalg.vector_norm(bounded, dim=-1).mean())

  low = 0.0
  high = 1.0
  while mean_norm(high) < target_mean_norm and high < 1.0e6:
    high *= 2.0
  for _ in range(24):
    middle = 0.5 * (low + high)
    if mean_norm(middle) < target_mean_norm:
      low = middle
    else:
      high = middle
  scale = 0.5 * (low + high)
  delta = scale * unit
  achieved = mean_norm(scale)
  return delta, {
    "gradient_norm": float(gradient_norm),
    "parameter_scale": scale,
    "target_mean_residual_norm": target_mean_norm,
    "calibration_mean_residual_norm": achieved,
  }


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
  parser.add_argument("--screen-seed", type=int, required=True)
  parser.add_argument("--screen-envs", type=int, default=64)
  parser.add_argument("--direction-seed-offset", type=int, default=124_000_000)
  parser.add_argument("--local-search", action="store_true")
  parser.add_argument("--parameter-sigma", type=float, default=0.02)
  parser.add_argument("--target-mean-residual-norm", type=float, default=0.001)
  parser.add_argument("--success-bonus", type=float, default=1.0)
  parser.add_argument("--minimum-heldout-cosine", type=float, default=0.05)
  parser.add_argument("--minimum-train-pairwise-cosine", type=float, default=0.0)
  parser.add_argument("--max-residual", type=float, default=0.10)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
  try:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v124 seeds must be comma-separated integers") from exc
  if len(seeds) != 4 or len(set(seeds)) != 4:
    raise ValueError("v124 requires exactly three train seeds and one held-out seed")
  return seeds


def _normalized_sha(value: str) -> str:
  normalized = value.strip().lower()
  if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
    raise ValueError("v124 checkpoint hash must be 64 hexadecimal digits")
  return normalized


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
  denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
  if float(denominator) <= 1.0e-12:
    return 0.0
  return float(torch.sum(first * second) / denominator)


def _collect_parameter_branch(
  runner,
  base_env,
  action_term,
  *,
  seed: int,
  direction_seed: int,
  sign: float,
  parameter_sigma: float,
  max_residual: float,
  geometry_sample_limit: int = 4096,
) -> dict[str, Any]:
  if sign not in (-1.0, 1.0):
    raise ValueError("v124 branch sign must be -1 or +1")
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
  generator = torch.Generator(device=base_env.device)
  generator.manual_seed(int(direction_seed))
  directions = torch.randn(
    (base_env.num_envs, ACTION_DIM, PARAMETER_WIDTH),
    generator=generator,
    device=base_env.device,
  ) / math.sqrt(float(PARAMETER_WIDTH))
  active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  success = torch.zeros_like(active)
  fell = torch.zeros_like(active)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  geometry_chunks: list[torch.Tensor] = []
  geometry_sample_count = 0
  residual_norm_sum = 0.0
  residual_count = 0
  actor = runner.alg.actor
  actor.eval()
  maximum_steps = int(base_env.max_episode_length) + 2
  with torch.no_grad():
    for _ in range(maximum_steps):
      base_action = actor(observations, stochastic_output=False)
      features = _features_from_observations(actor, observations)
      geometry = features[:, BASE_HIDDEN_DIM : BASE_HIDDEN_DIM + GEOMETRY_DIM]
      augmented = torch.cat(
        (geometry, torch.ones((len(geometry), 1), device=geometry.device)), dim=-1
      )
      raw = float(sign) * float(parameter_sigma) * torch.einsum(
        "ni,nai->na", augmented, directions
      )
      correction = float(max_residual) * torch.tanh(raw / float(max_residual))
      geometry_active = _geometry_active_from_observations(observations)
      correction = correction * geometry_active.unsqueeze(-1)
      next_observations, _, dones, extras = runner.env.step(base_action + correction)
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
        selected = ids[geometry_active[ids]]
        remaining = geometry_sample_limit - geometry_sample_count
        if selected.numel() and remaining > 0:
          sample = geometry[selected[:remaining]].cpu()
          geometry_chunks.append(sample)
          geometry_sample_count += len(sample)
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
  if bool(active.any()) or not geometry_chunks:
    raise RuntimeError("v124 parameter branch did not finish or lacks geometry")
  return {
    "seed": seed,
    "direction_seed": direction_seed,
    "sign": sign,
    "initial_state_signature": signature,
    "success": success.cpu(),
    "reached_risers": reached_risers.cpu(),
    "directions": directions.cpu(),
    "geometry_samples": torch.cat(geometry_chunks),
    "success_count": int(success.sum()),
    "fall_count": int(fell.sum()),
    "mean_reached_riser": float(reached_risers.float().mean()),
    "mean_perturbation_residual_norm": residual_norm_sum / max(1, residual_count),
  }


def main() -> None:
  args = _parse_args()
  seeds = _parse_seeds(args.training_seeds)
  method_id = parameter_method_id(args.local_search)
  if args.num_envs < 2 or not 1 <= args.screen_envs <= args.num_envs:
    raise ValueError("v124 environment counts are invalid")
  if not 0.001 <= args.parameter_sigma <= 0.1:
    raise ValueError("v124 parameter sigma is outside [0.001, 0.1]")
  if args.local_search and args.parameter_sigma > 0.01:
    raise ValueError("v125 local parameter sigma must not exceed 0.01")
  if not 0.0001 <= args.target_mean_residual_norm <= 0.01:
    raise ValueError("v124 target residual norm is outside [0.0001, 0.01]")
  if not 0.0 <= args.success_bonus <= 4.0:
    raise ValueError("v124 success bonus is outside [0, 4]")
  if not 0.01 <= args.max_residual <= 0.25:
    raise ValueError("v124 residual bound is outside [0.01, 0.25]")
  if not -1.0 <= args.minimum_train_pairwise_cosine <= 1.0:
    raise ValueError("v124 train cosine threshold is invalid")
  if not -1.0 <= args.minimum_heldout_cosine <= 1.0:
    raise ValueError("v124 held-out cosine threshold is invalid")

  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v124 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v124 checkpoint or search protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v124 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v124 base checkpoint SHA-256 differs")
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
      "parameter_sigma": args.parameter_sigma,
      "local_parameter_search": args.local_search,
      "target_mean_residual_norm": args.target_mean_residual_norm,
      "training_runtime_filter": False,
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
    raise RuntimeError("v124 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(action_term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v124 requires the current velocity-CBF action")

  try:
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_state, expansion = _expand_actor_state(
      source_payload["actor_state_dict"], runner.alg.actor.state_dict()
    )
    runner.alg.actor.load_state_dict(expanded_state, strict=True)
    runner.alg.actor.eval()
    for parameter in runner.alg.actor.parameters():
      parameter.requires_grad_(False)
    residual = LinearGeometryResidual(args.max_residual).to(args.device)
    initial_residual_state = actor_state(residual)
    initial_residual_sha = actor_state_sha256(initial_residual_state)

    seed_gradients: list[torch.Tensor] = []
    train_geometry: list[torch.Tensor] = []
    pair_summaries: list[dict[str, Any]] = []
    total_plus_success = total_minus_success = 0
    for seed_index, seed in enumerate(seeds):
      branches: list[dict[str, Any]] = []
      direction_seed = int(seed + args.direction_seed_offset)
      for sign in (1.0, -1.0):
        branch = _collect_parameter_branch(
          runner,
          base_env,
          action_term,
          seed=seed,
          direction_seed=direction_seed,
          sign=sign,
          parameter_sigma=args.parameter_sigma,
          max_residual=args.max_residual,
        )
        print(
          json.dumps(
            {
              "parameter_branch_completed": {
                key: value
                for key, value in branch.items()
                if key not in (
                  "success",
                  "reached_risers",
                  "directions",
                  "geometry_samples",
                )
              }
            }
          ),
          flush=True,
        )
        branches.append(branch)
      plus, minus = branches
      if plus["initial_state_signature"] != minus["initial_state_signature"]:
        raise RuntimeError("v124 mirrored initial-state signatures differ")
      if not torch.equal(plus["directions"], minus["directions"]):
        raise RuntimeError("v124 mirrored parameter directions differ")
      plus_scores = plus["reached_risers"].float() + float(
        args.success_bonus
      ) * plus["success"].float()
      minus_scores = minus["reached_risers"].float() + float(
        args.success_bonus
      ) * minus["success"].float()
      score_differences = plus_scores - minus_scores
      gradient, standardized_scores = standardized_parameter_gradient(
        plus["directions"], score_differences
      )
      seed_gradients.append(gradient)
      if seed_index < len(seeds) - 1:
        train_geometry.extend(
          (plus["geometry_samples"], minus["geometry_samples"])
        )
      total_plus_success += int(plus["success_count"])
      total_minus_success += int(minus["success_count"])
      pair_summaries.append(
        {
          "seed": seed,
          "direction_seed": direction_seed,
          "matched_initial_state_signature": plus["initial_state_signature"],
          "plus_success_count": plus["success_count"],
          "minus_success_count": minus["success_count"],
          "plus_mean_reached_riser": plus["mean_reached_riser"],
          "minus_mean_reached_riser": minus["mean_reached_riser"],
          "plus_mean_perturbation_residual_norm": plus[
            "mean_perturbation_residual_norm"
          ],
          "minus_mean_perturbation_residual_norm": minus[
            "mean_perturbation_residual_norm"
          ],
          "positive_difference_count": int((score_differences > 0).sum()),
          "negative_difference_count": int((score_differences < 0).sum()),
          "zero_difference_count": int((score_differences == 0).sum()),
          "score_difference_mean": float(score_differences.mean()),
          "score_difference_std": float(score_differences.std(unbiased=False)),
          "standardized_score_mean": float(standardized_scores.mean()),
          "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        }
      )

    train_gradient_stack = torch.stack(seed_gradients[:-1])
    train_gradient = train_gradient_stack.mean(dim=0)
    heldout_gradient = seed_gradients[-1]
    train_pairwise_cosines = [
      _cosine(train_gradient_stack[first], train_gradient_stack[second])
      for first in range(len(train_gradient_stack))
      for second in range(first + 1, len(train_gradient_stack))
    ]
    mean_train_pairwise_cosine = sum(train_pairwise_cosines) / len(
      train_pairwise_cosines
    )
    heldout_cosine = _cosine(train_gradient, heldout_gradient)
    offline_gate_passed = bool(
      mean_train_pairwise_cosine >= args.minimum_train_pairwise_cosine
      and heldout_cosine >= args.minimum_heldout_cosine
    )
    calibration = None
    if offline_gate_passed:
      delta, calibration = calibrated_parameter_delta(
        train_gradient,
        torch.cat(train_geometry),
        target_mean_norm=args.target_mean_residual_norm,
        max_residual=args.max_residual,
      )
      with torch.no_grad():
        residual.linear.weight.copy_(delta[:, :GEOMETRY_DIM].to(args.device))
        residual.linear.bias.copy_(delta[:, GEOMETRY_DIM].to(args.device))
    else:
      residual.load_state_dict(initial_residual_state, strict=True)
    final_residual_state = actor_state(residual)
    final_residual_sha = actor_state_sha256(final_residual_state)
    gradient_summary = {
      "train_seed_gradient_norms": [
        float(torch.linalg.vector_norm(value)) for value in train_gradient_stack
      ],
      "heldout_gradient_norm": float(torch.linalg.vector_norm(heldout_gradient)),
      "train_mean_gradient_norm": float(torch.linalg.vector_norm(train_gradient)),
      "train_pairwise_cosines": train_pairwise_cosines,
      "mean_train_pairwise_cosine": mean_train_pairwise_cosine,
      "heldout_gradient_cosine": heldout_cosine,
      "minimum_train_pairwise_cosine": args.minimum_train_pairwise_cosine,
      "minimum_heldout_cosine": args.minimum_heldout_cosine,
      "offline_gate_passed": offline_gate_passed,
    }

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
        "residual_state_dict": {
          key: value.detach().cpu() for key, value in final_residual_state.items()
        },
        "residual_state_sha256": final_residual_sha,
        "residual_config": {
          "type": "linear-persistent-geometry",
          "geometry_dim": GEOMETRY_DIM,
          "max_residual": args.max_residual,
          "parameter_sigma": args.parameter_sigma,
          "target_mean_residual_norm": args.target_mean_residual_norm,
        },
        "gradient_summary": gradient_summary,
        "calibration": calibration,
        "offline_gate_passed": offline_gate_passed,
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
      "initial_residual_sha256": initial_residual_sha,
      "final_residual_sha256": final_residual_sha,
      "trainable_parameter_count": sum(
        parameter.numel() for parameter in residual.parameters()
      ),
      "training_seeds": seeds,
      "training_seed_count": 3,
      "validation_seed": seeds[-1],
      "num_envs": args.num_envs,
      "training_runtime_filter": False,
      "parameter_sigma": args.parameter_sigma,
      "local_parameter_search": args.local_search,
      "target_mean_residual_norm": args.target_mean_residual_norm,
      "success_bonus": args.success_bonus,
      "plus_success_count": total_plus_success,
      "minus_success_count": total_minus_success,
      "pair_summaries": pair_summaries,
      "gradient_summary": gradient_summary,
      "calibration": calibration,
      "offline_gate_passed": offline_gate_passed,
      "transactional_rollback": not offline_gate_passed,
      "screen": screen,
      "independent_gate_run": False,
      "independent_gate_policy": "run separately only if screen_rate_gte_0.75",
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "actor_expansion": expansion,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
