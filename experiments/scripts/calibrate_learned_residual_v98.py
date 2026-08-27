"""Calibrate one learned v97 CBF-residual direction without retraining it."""

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
from refine_rescue_distill_v36 import (
  _atomic_json,
  _atomic_torch,
  _git,
  _initial_state_signature,
  _seed_everything,
)
from train_learned_residual_v97 import (
  METHOD_ID as SOURCE_METHOD_ID,
  SUCCESSFUL_EPISODE_METHOD_ID,
  TASK_METRIC_SUCCESSFUL_EPISODE_METHOD_ID,
  LearnedCbfResidual,
  _evaluate_filter_off,
  _policy_step,
  _set_filter,
  _write_csv,
)
from velocity_cbf_v34_protocol import CURRENT_CBF_MODE, PROTOCOL_ID


METHOD_ID = "heldout-group-scaled-learned-cbf-residual-v98"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--search-config", type=Path, required=True)
  parser.add_argument("--source-checkpoint", type=Path, required=True)
  parser.add_argument("--expected-source-sha256", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=tuple(CONTEXTS), required=True)
  parser.add_argument("--screen-scales", default="0,0.05,0.10,0.20")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--screen-seed", type=int, required=True)
  parser.add_argument("--gate-envs", type=int, default=64)
  parser.add_argument("--gate-seed", type=int, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _parse_scales(raw: str) -> list[float]:
  try:
    scales = [float(value.strip()) for value in raw.split(",") if value.strip()]
  except ValueError as exc:
    raise ValueError("v98 scales must be comma-separated numbers") from exc
  if len(scales) < 2 or len(set(scales)) != len(scales):
    raise ValueError("v98 requires at least two unique scales")
  if scales[0] != 0.0 or any(not 0.0 <= scale <= 1.0 for scale in scales):
    raise ValueError("v98 scales must start at zero and lie in [0, 1]")
  return scales


def _screen_scales(
  runner,
  base_env,
  action_term,
  residual,
  *,
  seed: int,
  scales: list[float],
) -> tuple[list[dict[str, Any]], float]:
  if base_env.num_envs % len(scales):
    raise ValueError("v98 num-envs must be divisible by the scale count")
  group_size = base_env.num_envs // len(scales)
  scale_by_env = torch.tensor(
    scales, device=base_env.device, dtype=torch.float32
  ).repeat_interleave(group_size)
  group_by_env = torch.arange(
    len(scales), device=base_env.device
  ).repeat_interleave(group_size)
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
  active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
  success = torch.zeros_like(active)
  fell = torch.zeros_like(active)
  steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
  reached_risers = torch.zeros_like(steps)
  residual_sum = torch.zeros(len(scales), device=base_env.device)
  transition_count = torch.zeros(
    len(scales), dtype=torch.long, device=base_env.device
  )
  counterfactual_count = torch.zeros_like(transition_count)
  maximum_steps = int(base_env.max_episode_length) + 2
  actor = runner.alg.actor
  actor.eval()
  residual.eval()
  with torch.no_grad():
    for _ in range(maximum_steps):
      actions, _, correction = _policy_step(
        actor,
        residual,
        observations,
        residual_scale=scale_by_env,
      )
      next_observations, _, dones, extras = runner.env.step(actions)
      extras = dict(extras)
      ids = active.nonzero(as_tuple=False).flatten()
      if ids.numel():
        reached_risers[ids] = torch.maximum(
          reached_risers[ids], extras["online_stair_index"][ids].long()
        )
        groups = group_by_env[ids]
        residual_sum.scatter_add_(
          0, groups, torch.linalg.vector_norm(correction[ids], dim=-1)
        )
        transition_count.scatter_add_(
          0, groups, torch.ones_like(groups, dtype=torch.long)
        )
        counterfactual_count.scatter_add_(
          0, groups, extras["cbf_would_intervene"][ids].long()
        )
      steps += active.long()
      completed = dones.bool() & active
      if bool(completed.any()):
        success_now = base_env.termination_manager.get_term("reached_top").bool()
        success[completed] = success_now[completed]
        fell[completed] = extras["online_fell"][completed].bool()
        active &= ~completed
        if not bool(active.any()):
          observations = next_observations
          break
      observations = next_observations
  if bool(active.any()):
    raise RuntimeError("v98 scale screen did not finish every first episode")
  summaries = []
  for group, scale in enumerate(scales):
    mask = group_by_env == group
    success_count = int(success[mask].sum())
    summaries.append(
      {
        "scale": scale,
        "screen_seed": seed,
        "initial_state_signature": signature,
        "episode_count": group_size,
        "success_count": success_count,
        "success_rate": success_count / group_size,
        "fall_count": int(fell[mask].sum()),
        "fall_rate": float(fell[mask].float().mean()),
        "mean_reached_riser": float(reached_risers[mask].float().mean()),
        "mean_residual_norm": float(
          residual_sum[group] / transition_count[group].clamp_min(1)
        ),
        "counterfactual_intervention_fraction": float(
          counterfactual_count[group] / transition_count[group].clamp_min(1)
        ),
      }
    )
  # Outcome first, progress second, and the smaller intervention last.
  selected = max(
    summaries,
    key=lambda item: (
      item["success_rate"],
      item["mean_reached_riser"],
      -item["scale"],
    ),
  )
  return summaries, float(selected["scale"])


def main() -> None:
  args = _parse_args()
  scales = _parse_scales(args.screen_scales)
  if args.num_envs < len(scales) or args.num_envs % len(scales):
    raise ValueError("v98 environment count must divide evenly across scales")
  if not 1 <= args.gate_envs <= args.num_envs:
    raise ValueError("v98 gate environment count is invalid")
  repo = args.repo.resolve()
  source_checkpoint = args.source_checkpoint.resolve()
  output = args.output_dir.resolve()
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v98 requires a clean committed worktree")
  if output.exists():
    raise FileExistsError(output)
  if not source_checkpoint.is_file() or not args.search_config.resolve().is_file():
    raise FileNotFoundError("v98 source checkpoint or protocol is missing")
  protocol = json.loads(args.search_config.resolve().read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("v98 velocity-CBF protocol differs")
  expected = args.expected_source_sha256.strip().lower()
  source_sha = file_sha256(source_checkpoint)
  if source_sha != expected:
    raise RuntimeError("v98 source checkpoint SHA-256 differs")
  source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
  if source.get("method_id") not in {
    SOURCE_METHOD_ID,
    SUCCESSFUL_EPISODE_METHOD_ID,
    TASK_METRIC_SUCCESSFUL_EPISODE_METHOD_ID,
  }:
    raise RuntimeError("v98 source method is not a supported learned residual")
  output.mkdir(parents=True)
  started = time.monotonic()
  source_commit = _git(repo, "rev-parse", "HEAD")

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
    TaskMetricVelocityCbfAction,
    configure_v34_cbf,
  )

  _seed_everything(args.screen_seed)
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
  source_cbf = source.get("training_cbf") or {
    "mode": CURRENT_CBF_MODE,
    "parameters": None,
  }
  cbf = configure_v34_cbf(
    env_cfg,
    mode=source_cbf["mode"],
    runtime_filter=False,
    parameters=source_cbf.get("parameters"),
    measure_compute_time=False,
  )
  reward = configure_paper_dual_reward(
    env_cfg, "raw_moderate", runtime_filter_during_training=True
  )
  geometry = configure_deployable_cbf_persistent_geometry_observation(env_cfg)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.screen_seed
  agent_cfg = load_rl_cfg(TASK_ID)
  configure_deployable_cbf_geometry_runner(agent_cfg)
  agent_cfg.seed = args.screen_seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("v98 task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  action_term = base_env.action_manager.get_term("joint_pos")
  if not isinstance(
    action_term,
    (InstrumentedCurrentVelocityCbfAction, TaskMetricVelocityCbfAction),
  ):
    raise TypeError("v98 requires a velocity-CBF action")
  try:
    runner.alg.actor.load_state_dict(source["base_actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    residual = LearnedCbfResidual(
      float(source["residual_config"]["max_residual"])
    ).to(args.device)
    residual.load_state_dict(source["residual_state_dict"], strict=True)
    residual.eval()
    actual_residual_hash = actor_state_sha256(actor_state(residual))
    if actual_residual_hash != source["residual_state_sha256"]:
      raise RuntimeError("v98 residual state hash differs")
    screen, selected_scale = _screen_scales(
      runner,
      base_env,
      action_term,
      residual,
      seed=args.screen_seed,
      scales=scales,
    )
    candidate_path = output / "candidate.pt"
    candidate = {
      **source,
      "schema_version": 2,
      "method_id": METHOD_ID,
      "v97_source_checkpoint_sha256": source_sha,
      "v98_git_commit": source_commit,
      "deployment_residual_scale": selected_scale,
      "scale_screen": screen,
    }
    _atomic_torch(candidate_path, candidate)
    gate, gate_rows = _evaluate_filter_off(
      runner,
      base_env,
      action_term,
      residual,
      seed=args.gate_seed,
      gate_envs=args.gate_envs,
      residual_scale=selected_scale,
    )
    gate["candidate_checkpoint_sha256"] = file_sha256(candidate_path)
    gate["residual_state_sha256"] = actual_residual_hash
    gate["passed_75_percent"] = gate["success_rate"] >= 0.75
    _atomic_json(output / "untouched_filter_off_gate.json", gate)
    _write_csv(output / "untouched_filter_off_gate.csv", gate_rows)
    summary = {
      "schema_version": 1,
      "method_id": METHOD_ID,
      "git_commit": source_commit,
      "context": args.context,
      "source_checkpoint": str(source_checkpoint),
      "source_checkpoint_sha256": source_sha,
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": file_sha256(candidate_path),
      "residual_state_sha256": actual_residual_hash,
      "screen_scales": scales,
      "screen_seed": args.screen_seed,
      "screen_episode_count_per_scale": args.num_envs // len(scales),
      "scale_screen": screen,
      "selected_scale": selected_scale,
      "selection_rule": "success_rate_then_mean_reached_riser_then_smaller_scale",
      "untouched_filter_off_gate": gate,
      "selected": gate["passed_75_percent"],
      "shift": shift,
      "cbf": cbf,
      "paper_dual_reward": reward,
      "geometry_observation": geometry,
      "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
