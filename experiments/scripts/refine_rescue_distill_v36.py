"""Distill CBF corrections only from matched filter-rescued initial episodes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
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
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID

METHOD_ID = "matched-filter-rescue-off-state-distillation-v36"
SHIELDED_METHOD_ID = "matched-filter-rescue-shielded-state-distillation-v37"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-base-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--teacher-eta", type=float, default=0.25)
  parser.add_argument("--actor-learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--moving-kl-beta", type=float, default=0.5)
  parser.add_argument(
    "--teacher-state-source", choices=("off", "on"), default="off"
  )
  parser.add_argument(
    "--max-reference-kl",
    type=float,
    default=0.0,
    help="Project the actor update to this dataset mean forward-KL; zero disables it.",
  )
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--minibatches", type=int, default=4)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _atomic_torch(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  torch.save(payload, temporary)
  temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _initial_state_signature(obs, base_env, action_term, command_term) -> str:
  signature = hashlib.sha256()
  terrain = base_env.scene.terrain
  if terrain is None:
    raise RuntimeError("v36 signature requires stair terrain")
  tensors = (
    obs["actor"],
    base_env.scene.env_origins,
    base_env.scene["robot"].data.root_link_pos_w,
    base_env.scene["robot"].data.root_link_quat_w,
    base_env.scene["robot"].data.root_link_lin_vel_w,
    base_env.scene["robot"].data.root_link_ang_vel_w,
    base_env.scene["robot"].data.joint_pos,
    base_env.scene["robot"].data.joint_vel,
    terrain.terrain_levels,
    terrain.terrain_types,
    action_term._edge_x[terrain.terrain_levels, terrain.terrain_types],
    action_term._edge_top_z[terrain.terrain_levels, terrain.terrain_types],
    base_env.command_manager.get_command("twist"),
    getattr(
      command_term,
      "raw_command",
      base_env.command_manager.get_command("twist"),
    ),
  )
  for tensor in tensors:
    signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return signature.hexdigest()


def _write_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _first_episode_rollout(
  *,
  repo: Path,
  checkpoint: Path,
  context: str,
  seed: int,
  num_envs: int,
  runtime_filter: bool,
  device: str,
  retain_runner: bool,
) -> tuple[dict[str, Any], Any | None, Any | None]:
  _seed_everything(seed)
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.velocity_cbf_action import (
    InstrumentedCurrentVelocityCbfAction,
    configure_v34_cbf,
  )

  env_cfg = load_env_cfg(TASK_ID, play=True)
  shift = configure_v31_context(
    env_cfg,
    context=context,
    runtime_filter=runtime_filter,
    context_spec=environment_parameters(context),
    clearance_barrier_slope=CLEARANCE_BARRIER_SLOPE,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  cbf = configure_v34_cbf(
    env_cfg,
    mode=CURRENT_CBF_MODE,
    runtime_filter=runtime_filter,
    parameters=None,
    measure_compute_time=False,
  )
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  agent_cfg = load_rl_cfg(TASK_ID)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v36 task has no inference runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
  policy = runner.get_inference_policy(device)
  base_env.seed(seed)
  obs, _ = env.reset()
  term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(term, InstrumentedCurrentVelocityCbfAction):
    raise TypeError("v36 requires the current velocity-CBF action")
  signature = _initial_state_signature(
    obs, base_env, term, base_env.command_manager.get_term("twist")
  )
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  successes = torch.zeros(num_envs, dtype=torch.bool, device=device)
  falls = torch.zeros_like(successes)
  steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  observations: list[torch.Tensor] = []
  nominal_actions: list[torch.Tensor] = []
  safe_actions: list[torch.Tensor] = []
  would_intervene: list[torch.Tensor] = []
  environment_ids: list[torch.Tensor] = []
  maximum_steps = int(base_env.max_episode_length) + 2
  with torch.inference_mode():
    for _ in range(maximum_steps):
      actor_observations = obs["actor"].detach()
      actions = policy(obs)
      obs, _, dones, extras = env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if retain_runner and ids.numel():
        observations.append(actor_observations[ids].cpu())
        nominal_actions.append(extras["cbf_nominal_raw_action"][ids].cpu())
        safe_actions.append(extras["cbf_safe_raw_action"][ids].cpu())
        would_intervene.append(extras["cbf_would_intervene"][ids].bool().cpu())
        environment_ids.append(ids.cpu())
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        success = base_env.termination_manager.get_term("reached_top").bool()
        fell = extras["online_fell"].bool()
        successes[completed] = success[completed]
        falls[completed] = fell[completed]
        active &= ~completed
        if not bool(active.any()):
          break
  if bool(active.any()):
    env.close()
    raise RuntimeError("v36 did not finish every first episode")
  result: dict[str, Any] = {
    "runtime_filter": runtime_filter,
    "initial_state_signature": signature,
    "actor_sha256": actor_hash,
    "success": successes.cpu(),
    "fell": falls.cpu(),
    "steps": steps.cpu(),
    "shift": shift,
    "cbf": cbf,
  }
  if retain_runner:
    result["dataset"] = {
      "observations": torch.cat(observations),
      "nominal_actions": torch.cat(nominal_actions),
      "safe_actions": torch.cat(safe_actions),
      "would_intervene": torch.cat(would_intervene),
      "environment_ids": torch.cat(environment_ids),
    }
    return result, runner, env
  env.close()
  return result, None, None


def _dataset_policy_metrics(
  actor,
  reference_actor,
  dataset: dict[str, torch.Tensor],
  eligible: torch.Tensor,
  weights: torch.Tensor,
  *,
  eta: float,
  device: str,
  batch_size: int,
) -> dict[str, float]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl

  weighted_distance = 0.0
  weighted_loss = 0.0
  weight_sum = 0.0
  kl_sum = 0.0
  mean_shift_sum = 0.0
  count = 0
  with torch.inference_mode():
    for start in range(0, len(eligible), batch_size):
      stop = min(len(eligible), start + batch_size)
      obs = dataset["observations"][start:stop].to(device)
      correction = (
        dataset["safe_actions"][start:stop]
        - dataset["nominal_actions"][start:stop]
      ).to(device)
      actor_obs = {"actor": obs}
      reference_actor(actor_obs, stochastic_output=True)
      reference_params = tuple(
        value.detach() for value in reference_actor.output_distribution_params
      )
      target = reference_params[0] + eta * correction
      actor(actor_obs, stochastic_output=True)
      current_params = tuple(actor.output_distribution_params)
      batch_weights = weights[start:stop].to(device)
      batch_eligible = eligible[start:stop].to(device)
      effective = batch_weights * batch_eligible.to(batch_weights.dtype)
      delta = current_params[0] - target
      per_transition_loss = F.smooth_l1_loss(
        current_params[0], target, reduction="none", beta=0.05
      ).mean(dim=-1)
      weighted_distance += float(
        (effective * torch.linalg.vector_norm(delta, dim=-1)).sum()
      )
      weighted_loss += float((effective * per_transition_loss).sum())
      weight_sum += float(effective.sum())
      kl = diagonal_gaussian_forward_kl(current_params, reference_params)
      kl_sum += float(kl.sum())
      mean_shift_sum += float(
        torch.linalg.vector_norm(
          current_params[0] - reference_params[0], dim=-1
        ).sum()
      )
      count += stop - start
  return {
    "teacher_weighted_distance": weighted_distance / max(1.0e-8, weight_sum),
    "teacher_weighted_smooth_l1": weighted_loss / max(1.0e-8, weight_sum),
    "reference_forward_kl": kl_sum / max(1, count),
    "reference_mean_shift": mean_shift_sum / max(1, count),
  }


def _distill_actor(
  actor,
  dataset: dict[str, torch.Tensor],
  rescued_env_ids: torch.Tensor,
  *,
  eta: float,
  learning_rate: float,
  moving_kl_beta: float,
  max_reference_kl: float,
  epochs: int,
  minibatches: int,
  max_grad_norm: float,
  device: str,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
  from src.tasks.stairs_cbf.proximal import diagonal_gaussian_forward_kl
  from src.tasks.stairs_cbf.teacher_v30_math import (
    weighted_smooth_l1_teacher_loss,
  )

  actor.eval()
  reference_actor = copy.deepcopy(actor).to(device).eval()
  for parameter in reference_actor.parameters():
    parameter.requires_grad_(False)
  rescued = torch.isin(dataset["environment_ids"], rescued_env_ids)
  eligible = rescued & dataset["would_intervene"].bool()
  correction = dataset["safe_actions"] - dataset["nominal_actions"]
  correction_norm = torch.linalg.vector_norm(correction, dim=-1)
  weights = eligible.float() * torch.clamp(correction_norm / 0.05, 0.0, 1.0)
  if not bool(eligible.any()):
    raise RuntimeError("v36 matched rescue dataset has no CBF intervention labels")
  parameters = list(actor.mlp.parameters())
  optimizer = torch.optim.Adam(parameters, lr=learning_rate)
  total = len(eligible)
  batch_size = math.ceil(total / minibatches)
  before = _dataset_policy_metrics(
    actor,
    reference_actor,
    dataset,
    eligible,
    weights,
    eta=eta,
    device=device,
    batch_size=batch_size,
  )
  updates = 0
  clipped = 0
  maximum_gradient_norm = 0.0
  teacher_loss_total = 0.0
  kl_loss_total = 0.0
  minibatches_with_teacher = 0
  for _ in range(epochs):
    permutation = torch.randperm(total)
    for start in range(0, total, batch_size):
      indices = permutation[start : start + batch_size]
      obs = dataset["observations"][indices].to(device)
      batch_correction = correction[indices].to(device)
      batch_eligible = eligible[indices].to(device)
      batch_weights = weights[indices].to(device)
      with torch.no_grad():
        actor_obs = {"actor": obs}
        reference_actor(actor_obs, stochastic_output=True)
        reference_params = tuple(
          value.detach() for value in reference_actor.output_distribution_params
        )
        target = reference_params[0] + eta * batch_correction
      actor(actor_obs, stochastic_output=True)
      current_params = tuple(actor.output_distribution_params)
      teacher_loss = weighted_smooth_l1_teacher_loss(
        current_params[0],
        target,
        batch_eligible,
        batch_weights,
        beta=0.05,
      )
      moving_kl = diagonal_gaussian_forward_kl(
        current_params, reference_params
      ).mean()
      loss = teacher_loss + moving_kl_beta * moving_kl
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
      if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("v36 actor gradient is non-finite")
      clipped += int(float(gradient_norm) > max_grad_norm)
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
      optimizer.step()
      updates += 1
      minibatches_with_teacher += int(bool(batch_eligible.any()))
      teacher_loss_total += float(teacher_loss.detach())
      kl_loss_total += float(moving_kl.detach())
  after = _dataset_policy_metrics(
    actor,
    reference_actor,
    dataset,
    eligible,
    weights,
    eta=eta,
    device=device,
    batch_size=batch_size,
  )
  unprojected_after = dict(after)
  interpolation_scale = 1.0
  projection_iterations = 0
  if max_reference_kl > 0.0 and after["reference_forward_kl"] > max_reference_kl:
    reference_state = {
      key: value.detach().clone()
      for key, value in reference_actor.mlp.state_dict().items()
    }
    proposal_state = {
      key: value.detach().clone() for key, value in actor.mlp.state_dict().items()
    }

    def load_interpolation(scale: float) -> None:
      actor.mlp.load_state_dict(
        {
          key: reference_state[key]
          + float(scale) * (proposal_state[key] - reference_state[key])
          for key in reference_state
        },
        strict=True,
      )

    low = 0.0
    high = 1.0
    for _ in range(12):
      projection_iterations += 1
      middle = 0.5 * (low + high)
      load_interpolation(middle)
      middle_metrics = _dataset_policy_metrics(
        actor,
        reference_actor,
        dataset,
        eligible,
        weights,
        eta=eta,
        device=device,
        batch_size=batch_size,
      )
      if middle_metrics["reference_forward_kl"] <= max_reference_kl:
        low = middle
      else:
        high = middle
    interpolation_scale = low
    load_interpolation(interpolation_scale)
    after = _dataset_policy_metrics(
      actor,
      reference_actor,
      dataset,
      eligible,
      weights,
      eta=eta,
      device=device,
      batch_size=batch_size,
    )
  return {
    "dataset_transition_count": total,
    "rescued_environment_count": int(rescued_env_ids.numel()),
    "rescued_transition_count": int(rescued.sum()),
    "teacher_transition_count": int(eligible.sum()),
    "teacher_transition_fraction": float(eligible.float().mean()),
    "teacher_weight_sum": float(weights.sum()),
    "teacher_eta": eta,
    "moving_kl_beta": moving_kl_beta,
    "epochs": epochs,
    "minibatches": minibatches,
    "optimizer_updates": updates,
    "minibatches_with_teacher": minibatches_with_teacher,
    "actor_gradient_clipped_fraction": clipped / max(1, updates),
    "actor_gradient_norm_pre_clip_max": maximum_gradient_norm,
    "teacher_loss_during_update": teacher_loss_total / max(1, updates),
    "moving_kl_during_update": kl_loss_total / max(1, updates),
    "before": before,
    "unprojected_after": unprojected_after,
    "after": after,
    "trust_region": {
      "enabled": max_reference_kl > 0.0,
      "max_reference_kl": max_reference_kl,
      "parameter_interpolation_scale": interpolation_scale,
      "projection_iterations": projection_iterations,
    },
  }, optimizer


def main() -> None:
  args = _parse_args()
  if args.num_envs < 2 or args.epochs not in (1, 2) or args.minibatches < 1:
    raise ValueError("v36 requires >=2 envs, one/two epochs, and minibatches")
  if not 0.0 < args.teacher_eta <= 1.0:
    raise ValueError("v36 teacher eta must lie in (0, 1]")
  if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-4:
    raise ValueError("v36 actor learning rate is outside the safe range")
  if not 0.0 <= args.moving_kl_beta <= 4.0 or args.max_grad_norm <= 0.0:
    raise ValueError("v36 KL beta or gradient norm is invalid")
  if not 0.0 <= args.max_reference_kl <= 0.05:
    raise ValueError("v36 reference-KL cap must lie in [0, 0.05]")
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  config_path = args.search_config.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v36 requires a clean committed worktree")
  if not checkpoint.is_file() or not config_path.is_file():
    raise FileNotFoundError("v36 checkpoint or search config is missing")
  config = json.loads(config_path.read_text())
  if config.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v36 velocity-CBF protocol differs")
  checkpoint_sha = file_sha256(checkpoint)
  if checkpoint_sha != args.expected_base_sha256.lower():
    raise RuntimeError("v36 base checkpoint SHA-256 differs")
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  started = time.monotonic()
  method_id = (
    SHIELDED_METHOD_ID if args.teacher_state_source == "on" else METHOD_ID
  )
  _atomic_json(
    output / "execution_started.json",
    {
      "method_id": method_id,
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "seed": args.seed,
      "base_checkpoint_sha256": checkpoint_sha,
    },
  )
  sys.path.insert(0, str(repo))
  from src.tasks.stairs_cbf.teacher_v30_math import filter_rescued_episode_mask

  if args.teacher_state_source == "off":
    on, _, _ = _first_episode_rollout(
      repo=repo,
      checkpoint=checkpoint,
      context=args.context,
      seed=args.seed,
      num_envs=args.num_envs,
      runtime_filter=True,
      device=args.device,
      retain_runner=False,
    )
    off, runner, env = _first_episode_rollout(
      repo=repo,
      checkpoint=checkpoint,
      context=args.context,
      seed=args.seed,
      num_envs=args.num_envs,
      runtime_filter=False,
      device=args.device,
      retain_runner=True,
    )
  else:
    off, _, _ = _first_episode_rollout(
      repo=repo,
      checkpoint=checkpoint,
      context=args.context,
      seed=args.seed,
      num_envs=args.num_envs,
      runtime_filter=False,
      device=args.device,
      retain_runner=False,
    )
    on, runner, env = _first_episode_rollout(
      repo=repo,
      checkpoint=checkpoint,
      context=args.context,
      seed=args.seed,
      num_envs=args.num_envs,
      runtime_filter=True,
      device=args.device,
      retain_runner=True,
    )
  if on["initial_state_signature"] != off["initial_state_signature"]:
    env.close()
    raise RuntimeError("v36 paired on/off initial states differ")
  if on["actor_sha256"] != off["actor_sha256"]:
    env.close()
    raise RuntimeError("v36 paired on/off actors differ")
  rescue_mask = filter_rescued_episode_mask(on["success"], off["success"])
  rescued_env_ids = rescue_mask.nonzero(as_tuple=False).flatten()
  if rescued_env_ids.numel() < 1:
    env.close()
    raise RuntimeError("v36 training seed contains no filter-rescued episodes")
  dataset = (on if args.teacher_state_source == "on" else off)["dataset"]
  dataset_payload = {
    "schema_version": 1,
    "method_id": method_id,
    "teacher_state_source": args.teacher_state_source,
    "seed": args.seed,
    "initial_state_signature": on["initial_state_signature"],
    "actor_sha256": on["actor_sha256"],
    "rescued_environment_ids": rescued_env_ids,
    **dataset,
  }
  dataset_path = output / "rescue_dataset.pt"
  _atomic_torch(dataset_path, dataset_payload)
  initial_actor_hash = actor_state_sha256(actor_state(runner.alg.actor))
  training, optimizer = _distill_actor(
    runner.alg.actor,
    dataset,
    rescued_env_ids,
    eta=args.teacher_eta,
    learning_rate=args.actor_learning_rate,
    moving_kl_beta=args.moving_kl_beta,
    max_reference_kl=args.max_reference_kl,
    epochs=args.epochs,
    minibatches=args.minibatches,
    max_grad_norm=args.max_grad_norm,
    device=args.device,
  )
  final_actor_state = actor_state(runner.alg.actor)
  final_actor_hash = actor_state_sha256(final_actor_state)
  source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  source_payload["actor_state_dict"] = {
    key: value.cpu() for key, value in final_actor_state.items()
  }
  source_payload["rescue_distill_optimizer_state_dict"] = optimizer.state_dict()
  source_payload["iter"] = int(source_payload.get("iter", 0)) + args.epochs
  infos = dict(source_payload.get("infos") or {})
  info_key = (
    "rescue_shielded_distill_v37"
    if args.teacher_state_source == "on"
    else "rescue_distill_v36"
  )
  infos[info_key] = {
    "method_id": method_id,
    "source_git_commit": _git(repo, "rev-parse", "HEAD"),
    "training_seed": args.seed,
    "teacher_eta": args.teacher_eta,
    "moving_kl_beta": args.moving_kl_beta,
    "teacher_state_source": args.teacher_state_source,
    "max_reference_kl": args.max_reference_kl,
    "rescued_environment_ids": rescued_env_ids.tolist(),
  }
  source_payload["infos"] = infos
  candidate_path = output / "candidate.pt"
  _atomic_torch(candidate_path, source_payload)
  rows = []
  for env_id in range(args.num_envs):
    rows.append(
      {
        "environment_id": env_id,
        "filter_on_success": bool(on["success"][env_id]),
        "filter_off_success": bool(off["success"][env_id]),
        "filter_rescued": bool(rescue_mask[env_id]),
        "filter_on_fell": bool(on["fell"][env_id]),
        "filter_off_fell": bool(off["fell"][env_id]),
        "filter_on_steps": int(on["steps"][env_id]),
        "filter_off_steps": int(off["steps"][env_id]),
      }
    )
  _write_outcomes(output / "paired_outcomes.csv", rows)
  summary = {
    "schema_version": 1,
    "method_id": method_id,
    "git_commit": _git(repo, "rev-parse", "HEAD"),
    "context": args.context,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "initial_state_signature": on["initial_state_signature"],
    "paired_actor_sha256": on["actor_sha256"],
    "base_checkpoint_sha256": checkpoint_sha,
    "filter_on_success_count": int(on["success"].sum()),
    "filter_off_success_count": int(off["success"].sum()),
    "filter_rescued_count": int(rescue_mask.sum()),
    "filter_rescued_environment_ids": rescued_env_ids.tolist(),
    "filter_harmed_count": int((~on["success"] & off["success"]).sum()),
    "dataset_path": str(dataset_path),
    "dataset_sha256": file_sha256(dataset_path),
    "candidate_path": str(candidate_path),
    "candidate_checkpoint_sha256": file_sha256(candidate_path),
    "initial_actor_sha256": initial_actor_hash,
    "candidate_actor_sha256": final_actor_hash,
    "training": training,
    "teacher_eta": args.teacher_eta,
    "actor_learning_rate": args.actor_learning_rate,
    "moving_kl_beta": args.moving_kl_beta,
    "teacher_state_source": args.teacher_state_source,
    "max_reference_kl": args.max_reference_kl,
    "epochs": args.epochs,
    "minibatches": args.minibatches,
    "elapsed_seconds": time.monotonic() - started,
    "evaluation_seed_separate_from_training": True,
  }
  _atomic_json(output / "training_summary.json", summary)
  env.close()
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
