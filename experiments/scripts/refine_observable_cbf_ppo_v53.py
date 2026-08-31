"""Multi-round on-policy CBF PPO for the deployable geometry adapter.

The paper performs a repeated collect-update loop, whereas v51/v52 took one
actor step from one frozen data set.  v53 refreshes paired filter-on/off
rollouts after every accepted actor update and fits the expanded critic before
the next round.  Only the five appended actor input columns remain trainable;
the legacy 405-D policy is an exact immutable fallback when geometry is idle.
"""

from __future__ import annotations

import argparse
import copy
import json
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
from refine_multi_rollout_gae_v46 import _collect_rollout
from refine_observable_cbf_adapter_v49 import _expand_actor_state
from refine_observable_cbf_ppo_v51 import (
  _expand_critic_state,
  _normalized_sha,
  _parse_seeds,
  _robust_adapter_ppo_step,
)
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _seed_everything,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "observable-cbf-geometry-multiround-paired-dual-gae-v53"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--training-seeds", required=True)
  parser.add_argument("--rounds", type=int, default=4)
  parser.add_argument("--round-seed-stride", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rollout-steps", type=int, default=512)
  parser.add_argument("--optimization-seed", type=int, required=True)
  parser.add_argument("--actor-learning-rate", type=float, default=0.01)
  parser.add_argument("--critic-learning-rate", type=float, default=1.0e-5)
  parser.add_argument("--critic-epochs", type=int, default=2)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument("--max-reference-kl", type=float, default=2.5e-5)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument("--max-grad-norm", type=float, default=10.0)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _critic_metrics(critic, batches: list[Any]) -> dict[str, float]:
  squared_error_sum = 0.0
  target_sum = 0.0
  target_square_sum = 0.0
  residual_sum = 0.0
  residual_square_sum = 0.0
  count = 0
  with torch.inference_mode():
    for batch in batches:
      targets = batch.returns.flatten(0, 1).squeeze(-1)
      values = critic(batch.observations.flatten(0, 1)).squeeze(-1)
      residuals = targets - values
      squared_error_sum += float(residuals.square().sum())
      target_sum += float(targets.sum())
      target_square_sum += float(targets.square().sum())
      residual_sum += float(residuals.sum())
      residual_square_sum += float(residuals.square().sum())
      count += targets.numel()
  target_variance = target_square_sum / count - (target_sum / count) ** 2
  residual_variance = (
    residual_square_sum / count - (residual_sum / count) ** 2
  )
  return {
    "mean_squared_error": squared_error_sum / count,
    "explained_variance": 1.0
    - residual_variance / max(target_variance, 1.0e-8),
    "transition_count": count,
  }


def _fit_critic(
  critic,
  batches: list[Any],
  *,
  learning_rate: float,
  epochs: int,
  max_grad_norm: float,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  parameters = tuple(critic.parameters())
  for parameter in parameters:
    parameter.requires_grad_(True)
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  critic.eval()
  before = _critic_metrics(critic, batches)
  critic.train()
  epoch_losses: list[float] = []
  gradient_norms: list[float] = []
  for _ in range(epochs):
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = 0.0
    for batch in batches:
      targets = batch.returns.flatten(0, 1).squeeze(-1).detach()
      values = critic(batch.observations.flatten(0, 1)).squeeze(-1)
      loss = (values - targets).square().mean()
      (loss / len(batches)).backward()
      epoch_loss += float(loss.detach()) / len(batches)
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
      raise RuntimeError("v53 critic gradient is non-finite")
    optimizer.step()
    epoch_losses.append(epoch_loss)
    gradient_norms.append(float(gradient_norm))
  critic.eval()
  after = _critic_metrics(critic, batches)
  return {
    "optimizer": "adam",
    "learning_rate": learning_rate,
    "epochs": epochs,
    "epoch_losses": epoch_losses,
    "gradient_norm_pre_clip": gradient_norms,
    "before": before,
    "after": after,
  }, optimizer


def _round_seeds(base_seeds: list[int], stride: int, round_index: int) -> list[int]:
  return [seed + stride * round_index for seed in base_seeds]


def main() -> None:
  args = _parse_args()
  base_seeds = _parse_seeds(args.training_seeds)
  if not 1 <= args.rounds <= 16:
    raise ValueError("v53 rounds must lie in [1, 16]")
  if args.round_seed_stride < 1:
    raise ValueError("v53 round seed stride must be positive")
  all_round_seeds = [
    seed
    for round_index in range(args.rounds)
    for seed in _round_seeds(base_seeds, args.round_seed_stride, round_index)
  ]
  if len(set(all_round_seeds)) != len(all_round_seeds):
    raise ValueError("v53 generated duplicate round seeds")
  if args.num_envs < 2 or args.rollout_steps < 64:
    raise ValueError("v53 rollout dimensions are too small")
  if not 1.0e-5 <= args.actor_learning_rate <= 0.1:
    raise ValueError("v53 actor learning rate is outside the supported range")
  if not 1.0e-7 <= args.critic_learning_rate <= 1.0e-3:
    raise ValueError("v53 critic learning rate is outside the supported range")
  if not 1 <= args.critic_epochs <= 8:
    raise ValueError("v53 critic epochs must lie in [1, 8]")
  if not 0.0 <= args.moving_kl_beta <= 4.0:
    raise ValueError("v53 moving KL beta must lie in [0, 4]")
  if not 0.0 < args.max_reference_kl <= 0.001:
    raise ValueError("v53 per-round reference KL cap must lie in (0, 0.001]")

  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  search_config = args.search_config.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v53 requires a clean committed worktree")
  if not checkpoint.is_file() or not search_config.is_file():
    raise FileNotFoundError("v53 checkpoint or search configuration is missing")
  protocol = json.loads(search_config.read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v53 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != _normalized_sha(args.expected_base_sha256):
    raise RuntimeError("v53 base checkpoint SHA-256 differs")
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  started = time.monotonic()
  git_commit = _git(repo, "rev-parse", "HEAD")
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": METHOD_ID,
      "git_commit": git_commit,
      "base_checkpoint_sha256": checkpoint_sha,
      "base_training_seeds": base_seeds,
      "all_round_seeds": all_round_seeds,
      "rounds": args.rounds,
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

  _seed_everything(base_seeds[0])
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
  env_cfg.seed = base_seeds[0]
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = base_seeds[0]
  agent_cfg.num_steps_per_env = args.rollout_steps
  _configure_algorithm(agent_cfg, "A0", preflight=False)
  agent_cfg.algorithm.minimum_std = 0.05
  agent_cfg.algorithm.maximum_std = 0.05
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner = CbfTeacherV30Runner(
    env, asdict(agent_cfg), log_dir=None, device=args.device
  )

  round_records: list[dict[str, Any]] = []
  accepted_rounds = 0
  last_actor_optimizer: torch.optim.Optimizer | None = None
  last_critic_optimizer: torch.optim.Optimizer | None = None
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
    initial_actor_state = actor_state(runner.alg.actor)
    initial_actor_hash = actor_state_sha256(initial_actor_state)
    action_term = base_env.action_manager.get_term("joint_pos")

    for round_index in range(args.rounds):
      round_number = round_index + 1
      round_started = time.monotonic()
      seeds = _round_seeds(base_seeds, args.round_seed_stride, round_index)
      runner.alg.freeze_round_reference()
      actor_before = copy.deepcopy(runner.alg.actor.state_dict())
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
            raise RuntimeError("v53 paired initial states differ")
          batches.append(batch)
          rollout_summaries.append(rollout)
      _seed_everything(args.optimization_seed + round_index)
      actor_training, actor_optimizer = _robust_adapter_ppo_step(
        runner.alg.actor,
        batches,
        rollout_summaries,
        learning_rate=args.actor_learning_rate,
        moving_kl_beta=args.moving_kl_beta,
        max_reference_kl=args.max_reference_kl,
        target_reference_kl=None,
        max_adapter_scale=1.0,
        line_search_iterations=12,
        clip_ratio=args.clip_ratio,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
      )
      after = actor_training["after"]
      actor_gate = (
        after["positive_batch_count"] == after["batch_count"]
        and after["mean_filter_on_surrogate_gain"] > 0.0
        and after["mean_filter_off_surrogate_gain"] > 0.0
        and after["reference_forward_kl"] <= args.max_reference_kl
        and actor_training["legacy_first_layer_change_max_abs"] == 0.0
      )
      if not actor_gate:
        runner.alg.actor.load_state_dict(actor_before, strict=True)
        round_records.append(
          {
            "round": round_number,
            "status": "rejected_offline_actor_gate",
            "training_seeds": seeds,
            "rollout_summaries": rollout_summaries,
            "actor_training": actor_training,
            "elapsed_seconds": time.monotonic() - round_started,
          }
        )
        print(json.dumps(round_records[-1], sort_keys=True), flush=True)
        break

      critic_training, critic_optimizer = _fit_critic(
        runner.alg.critic,
        batches,
        learning_rate=args.critic_learning_rate,
        epochs=args.critic_epochs,
        max_grad_norm=args.max_grad_norm,
      )
      accepted_rounds += 1
      last_actor_optimizer = actor_optimizer
      last_critic_optimizer = critic_optimizer
      round_record = {
        "round": round_number,
        "status": "accepted",
        "training_seeds": seeds,
        "rollout_summaries": rollout_summaries,
        "actor_training": actor_training,
        "critic_training": critic_training,
        "actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
        "elapsed_seconds": time.monotonic() - round_started,
      }
      round_records.append(round_record)
      _atomic_json(output / "round_metrics.json", round_records)
      print(json.dumps(round_record, sort_keys=True), flush=True)

    final_actor_state = actor_state(runner.alg.actor)
    final_first_layer = final_actor_state["mlp.0.weight"]
    initial_first_layer = initial_actor_state["mlp.0.weight"]
    legacy_change = float(
      (
        final_first_layer[:, : initial_first_layer.shape[1] - 5]
        - initial_first_layer[:, : initial_first_layer.shape[1] - 5]
      )
      .abs()
      .max()
    )
    complete_training_gate = accepted_rounds == args.rounds
    candidate_payload = copy.deepcopy(source_payload)
    candidate_payload["actor_state_dict"] = {
      key: value.detach().cpu() for key, value in final_actor_state.items()
    }
    candidate_payload["critic_state_dict"] = {
      key: value.detach().cpu()
      for key, value in runner.alg.critic.state_dict().items()
    }
    candidate_payload["actor_observation_interface"] = (
      "legacy_405_plus_deployable_cbf_geometry_5"
    )
    if last_actor_optimizer is not None:
      candidate_payload["observable_cbf_ppo_optimizer_state_dict"] = (
        last_actor_optimizer.state_dict()
      )
    if last_critic_optimizer is not None:
      candidate_payload["observable_cbf_critic_optimizer_state_dict"] = (
        last_critic_optimizer.state_dict()
      )
    infos = dict(candidate_payload.get("infos") or {})
    infos[METHOD_ID] = {
      "source_git_commit": git_commit,
      "base_training_seeds": base_seeds,
      "round_seed_stride": args.round_seed_stride,
      "accepted_rounds": accepted_rounds,
      "requested_rounds": args.rounds,
      "offline_gate_passed": complete_training_gate,
    }
    candidate_payload["infos"] = infos
    candidate_path = output / "candidate.pt"
    _atomic_torch(candidate_path, candidate_payload)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": git_commit,
      "context": args.context,
      "base_training_seeds": base_seeds,
      "round_seed_stride": args.round_seed_stride,
      "all_round_seeds": all_round_seeds,
      "requested_rounds": args.rounds,
      "accepted_rounds": accepted_rounds,
      "num_envs": args.num_envs,
      "rollout_steps": args.rollout_steps,
      "training_transition_count": accepted_rounds
      * len(base_seeds)
      * 2
      * args.num_envs
      * args.rollout_steps,
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
      "candidate_actor_sha256": actor_state_sha256(final_actor_state),
      "actor_learning_rate": args.actor_learning_rate,
      "critic_learning_rate": args.critic_learning_rate,
      "critic_epochs": args.critic_epochs,
      "moving_kl_beta": args.moving_kl_beta,
      "max_reference_kl_per_round": args.max_reference_kl,
      "clip_ratio": args.clip_ratio,
      "legacy_first_layer_change_max_abs": legacy_change,
      "inactive_geometry_exact_base_policy": legacy_change == 0.0,
      "offline_gate_passed": complete_training_gate and legacy_change == 0.0,
      "rounds": round_records,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
