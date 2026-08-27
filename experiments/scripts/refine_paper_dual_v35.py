"""Train one short paper-aligned dual CBF-RL reward candidate."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cbf_teacher_v31_protocol import (
  BASE_CHECKPOINT_SHA256,
  CLEARANCE_BARRIER_SLOPE,
  FILTER_ALPHA,
  RECOVERY_DISTANCE_M,
  TASK_ID,
  arm_parameters,
  environment_parameters,
)
from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from refine_cbf_teacher_v31 import (
  _collect_round,
  _configure_algorithm,
  _save_checkpoint,
  _write_round_csv,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument(
    "--expected-base-sha256",
    default=BASE_CHECKPOINT_SHA256,
    help=(
      "Exact SHA-256 required for the base checkpoint. Override this only for "
      "an explicitly recorded continuation checkpoint."
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=("F1", "F2", "F3"), required=True)
  parser.add_argument(
    "--candidate",
    choices=(
      "current",
      "raw_moderate",
      "raw_strong",
      "raw_demo",
      "paper_stair_exact",
      "paper_stair_demo_scale",
    ),
    required=True,
  )
  parser.add_argument(
    "--clearance-barrier-slope",
    type=float,
    default=CLEARANCE_BARRIER_SLOPE,
    help="Use 0 for the paper's horizontal next-riser hyperplane.",
  )
  parser.add_argument("--teacher-arm", choices=("A0", "A1", "A2"), default="A0")
  parser.add_argument(
    "--teacher-schedule",
    choices=("fixed", "A2_then_A1"),
    default="fixed",
    help="Use one fixed arm or switch from A2 residual to A1 full-action.",
  )
  parser.add_argument(
    "--teacher-switch-after",
    type=int,
    default=4,
    help="Last A2 round for the A2_then_A1 schedule.",
  )
  parser.add_argument(
    "--a1-teacher-weight",
    type=float,
    default=0.1,
    help="Full-action A1 loss weight; 0.1 preserves the frozen v31 arm.",
  )
  parser.add_argument(
    "--a2-teacher-eta",
    type=float,
    choices=(0.25, 0.5, 1.0),
    default=0.25,
    help=(
      "Fraction of the actor-coordinate CBF correction used by the bounded "
      "A2 Smooth-L1 target."
    ),
  )
  parser.add_argument(
    "--deterministic-mean-teacher",
    action="store_true",
    help=(
      "Project the frozen deterministic policy mean at the rollout state and "
      "use that counterfactual safe mean as the A2 target."
    ),
  )
  parser.add_argument(
    "--failure-only-mean-teacher",
    action="store_true",
    help=(
      "Gate deterministic-mean CBF labels to transitions from unshielded "
      "episodes that actually end in a fall."
    ),
  )
  parser.add_argument(
    "--success-only-mean-teacher",
    action="store_true",
    help=(
      "Gate deterministic-mean CBF labels to complete reached-top episodes "
      "from a shielded rollout."
    ),
  )
  parser.add_argument(
    "--failure-focused-actor",
    action="store_true",
    help=(
      "Use PPO and entropy actor gradients only on complete failed episodes; "
      "successful episodes retain only the moving round-reference KL."
    ),
  )
  parser.add_argument(
    "--distill-only-actor",
    action="store_true",
    help=(
      "Disable PPO/entropy actor gradients and update the actor only with "
      "the mean-CBF teacher plus the global moving reference KL."
    ),
  )
  parser.add_argument(
    "--success-local-kl-beta",
    type=float,
    default=0.0,
    help=(
      "Additional round-reference forward-KL coefficient on transitions from "
      "complete reached-top episodes."
    ),
  )
  parser.add_argument(
    "--height-curriculum",
    action="store_true",
    help="Train uniform contexts on ordered stair heights up to the target.",
  )
  parser.add_argument("--curriculum-start-height", type=float, default=0.13)
  parser.add_argument("--curriculum-rows", type=int, default=5)
  parser.add_argument("--rounds", type=int, default=8)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=1024)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--training-runtime-filter",
    choices=("on", "off"),
    default="on",
    help=(
      "Execute the CBF projection during rollout, or execute nominal actions "
      "while retaining the counterfactual CBF dual reward."
    ),
  )
  parser.add_argument(
    "--training-filter-fraction",
    type=float,
    default=None,
    help=(
      "Fraction of vector environments that execute the runtime filter in "
      "each round; defaults to 1 for on and 0 for off."
    ),
  )
  parser.add_argument(
    "--training-action-std",
    type=float,
    default=0.05,
    help="Fixed stochastic rollout std; evaluation remains deterministic.",
  )
  parser.add_argument(
    "--actor-learning-rate",
    type=float,
    default=5.0e-6,
    help="Actor learning rate recorded for the v35 continuation.",
  )
  parser.add_argument(
    "--moving-kl-beta",
    type=float,
    default=0.5,
    help=(
      "Extra round-reference KL coefficient. Use 0 for paper-standard clipped "
      "PPO without the historical continuation anchor."
    ),
  )
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _teacher_arms_by_round(
  *,
  rounds: int,
  teacher_arm: str,
  teacher_schedule: str,
  switch_after: int,
) -> list[str]:
  """Resolve the teacher arm before any environment or GPU allocation."""
  if rounds < 1:
    raise ValueError("v35 requires at least one round")
  if teacher_schedule == "fixed":
    arm_parameters(teacher_arm)
    return [teacher_arm] * rounds
  if teacher_schedule != "A2_then_A1":
    raise ValueError(f"unknown v35 teacher schedule {teacher_schedule!r}")
  if not 1 <= switch_after < rounds:
    raise ValueError("A2_then_A1 switch must leave at least one round for each arm")
  return ["A2"] * switch_after + ["A1"] * (rounds - switch_after)


def _set_teacher_arm(
  algorithm,
  arm: str,
  *,
  a1_teacher_weight: float,
  a2_teacher_eta: float,
) -> dict[str, Any]:
  """Switch the four teacher fields at a recorded round boundary."""
  parameters = arm_parameters(arm)
  if arm == "A1":
    parameters["teacher_weight"] = float(a1_teacher_weight)
    parameters["name"] = (
      f"full_action_local_success_50_weight_{a1_teacher_weight:g}"
    )
  elif arm == "A2":
    parameters["teacher_eta"] = float(a2_teacher_eta)
    parameters["name"] = (
      f"residual_eta_{a2_teacher_eta:g}_all_interventions"
    )
  algorithm.teacher_mode = parameters["teacher_mode"]
  algorithm.teacher_gate = parameters["teacher_gate"]
  algorithm.teacher_eta = parameters["teacher_eta"]
  algorithm.teacher_distillation_weight = parameters["teacher_weight"]
  return parameters


def _configure_height_curriculum(
  env_cfg,
  shift: dict[str, Any],
  *,
  start_height: float,
  num_rows: int,
) -> dict[str, Any]:
  """Turn one fixed uniform target into an ordered training curriculum."""
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

  from src.tasks.velocity.mdp.curriculums import terrain_levels_vel

  target_height = shift.get("riser_height_m")
  if target_height is None or shift.get("riser_profile_m") is not None:
    raise ValueError("v35 height curriculum currently requires a uniform context")
  start = float(start_height)
  target = float(target_height)
  if not math.isfinite(start) or not 0.02 <= start < target:
    raise ValueError("curriculum start height must lie in [0.02, target)")
  if num_rows < 2:
    raise ValueError("height curriculum requires at least two terrain rows")
  terrain_cfg = env_cfg.scene.terrain
  generator = None if terrain_cfg is None else terrain_cfg.terrain_generator
  if generator is None or set(generator.sub_terrains) != {"forward_stairs"}:
    raise RuntimeError("v35 curriculum requires one forward-stair terrain")
  stairs = generator.sub_terrains["forward_stairs"]
  if stairs.step_height_profile is not None:
    raise RuntimeError("v35 curriculum cannot vary an explicit riser profile")
  generator.sub_terrains["forward_stairs"] = replace(
    stairs,
    step_height_range=(start, target),
    step_height_profile=None,
  )
  generator.curriculum = True
  generator.num_rows = int(num_rows)
  terrain_cfg.max_init_terrain_level = 0
  env_cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=terrain_levels_vel,
      params={"command_name": "twist"},
    )
  }
  return {
    "enabled": True,
    "start_height_m": start,
    "target_height_m": target,
    "num_rows": int(num_rows),
    "initial_level": 0,
    "promotion_rule": "terrain_levels_vel",
    "exact_per_level_cbf_geometry": True,
    "target_evaluation_remains_fixed": True,
  }


def _terrain_level_metrics(base_env, *, num_rows: int) -> dict[str, Any]:
  terrain = base_env.scene.terrain
  if terrain is None or terrain.terrain_levels is None:
    raise RuntimeError("height curriculum terrain levels are unavailable")
  levels = terrain.terrain_levels.detach()
  histogram = torch.bincount(levels, minlength=num_rows)
  return {
    "terrain_level_mean": float(levels.float().mean()),
    "terrain_level_min": int(levels.min()),
    "terrain_level_max": int(levels.max()),
    "terrain_level_histogram": [int(value) for value in histogram.tolist()],
  }


def main() -> None:
  args = _parse_args()
  if args.rounds < 1 or args.num_envs < 1 or args.rollout_steps < 1:
    raise ValueError("v35 rounds, environments, and rollout steps must be positive")
  if not 0.0 <= args.clearance_barrier_slope <= 2.0:
    raise ValueError("v35 clearance barrier slope must lie in [0, 2]")
  if (
    args.candidate in {"paper_stair_exact", "paper_stair_demo_scale"}
    and args.clearance_barrier_slope != 0.0
  ):
    raise ValueError("paper stair candidates require the horizontal barrier (slope 0)")
  if not 0.0 < args.a1_teacher_weight <= 0.1:
    raise ValueError("v35 A1 teacher weight must be in (0, 0.1]")
  if args.curriculum_rows < 2:
    raise ValueError("v35 curriculum rows must be at least two")
  if not 0.01 <= args.training_action_std <= 0.05:
    raise ValueError("v35 training action std must lie in [0.01, 0.05]")
  if not 1.0e-7 <= args.actor_learning_rate <= 5.0e-6:
    raise ValueError("v35 actor learning rate must lie in [1e-7, 5e-6]")
  if not 0.0 <= args.moving_kl_beta <= 0.5:
    raise ValueError("v35 moving KL beta must lie in [0, 0.5]")
  training_runtime_filter = args.training_runtime_filter == "on"
  training_filter_fraction = (
    1.0 if training_runtime_filter else 0.0
  ) if args.training_filter_fraction is None else float(
    args.training_filter_fraction
  )
  if training_runtime_filter:
    if not 0.0 < training_filter_fraction <= 1.0:
      raise ValueError("enabled training filter fraction must lie in (0, 1]")
  elif training_filter_fraction != 0.0:
    raise ValueError("disabled training filter requires fraction 0")
  teacher_arms = _teacher_arms_by_round(
    rounds=args.rounds,
    teacher_arm=args.teacher_arm,
    teacher_schedule=args.teacher_schedule,
    switch_after=args.teacher_switch_after,
  )
  if args.deterministic_mean_teacher and any(arm != "A2" for arm in teacher_arms):
    raise ValueError("v35 deterministic-mean teacher currently requires only A2")
  if args.failure_only_mean_teacher and not args.deterministic_mean_teacher:
    raise ValueError("failure-only mean teacher requires deterministic mean labels")
  if args.failure_only_mean_teacher and args.training_runtime_filter != "off":
    raise ValueError("failure-only mean teacher requires unshielded training")
  if args.success_only_mean_teacher and not args.deterministic_mean_teacher:
    raise ValueError("success-only mean teacher requires deterministic mean labels")
  if args.failure_only_mean_teacher and args.success_only_mean_teacher:
    raise ValueError("mean-teacher outcome gates are mutually exclusive")
  if args.success_only_mean_teacher and training_filter_fraction != 1.0:
    raise ValueError("success-only mean teacher requires fully shielded training")
  if args.failure_focused_actor and not args.failure_only_mean_teacher:
    raise ValueError(
      "failure-focused actor requires the failure-only mean teacher"
    )
  if args.distill_only_actor and not args.deterministic_mean_teacher:
    raise ValueError("distillation-only actor requires deterministic mean labels")
  if not 0.0 <= args.success_local_kl_beta <= 4.0:
    raise ValueError("success-local KL beta must lie in [0, 4]")
  if args.success_local_kl_beta > 0.0 and (
    not args.deterministic_mean_teacher
    or args.failure_only_mean_teacher
    or args.success_only_mean_teacher
    or args.failure_focused_actor
    or args.distill_only_actor
    or args.training_runtime_filter != "off"
  ):
    raise ValueError(
      "success-local KL requires unshielded all-intervention mean-CBF training"
    )
  if (
    args.training_runtime_filter == "off"
    and not args.deterministic_mean_teacher
    and any(arm != "A0" for arm in teacher_arms)
  ):
    raise ValueError(
      "unshielded v35 training requires A0 unless the explicit "
      "counterfactual deterministic-mean teacher is enabled"
    )
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output_dir = args.output_dir.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  expected_base_sha256 = args.expected_base_sha256.strip().lower()
  if len(expected_base_sha256) != 64 or any(
    character not in "0123456789abcdef" for character in expected_base_sha256
  ):
    raise ValueError("v35 expected base SHA-256 must contain 64 hex digits")
  checkpoint_sha256 = file_sha256(checkpoint)
  if checkpoint_sha256 != expected_base_sha256:
    raise RuntimeError(
      "v35 base checkpoint does not match the explicitly expected SHA-256: "
      f"{checkpoint_sha256} != {expected_base_sha256}"
    )
  if output_dir.exists():
    raise FileExistsError(output_dir)
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("v35 training requires a clean committed worktree")

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.environment_v31 import configure_v31_context
  from src.tasks.stairs_cbf.paper_dual_v35 import configure_paper_dual_reward
  from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner
  from src.tasks.stairs_cbf.teacher_v30_math import (
    rotating_environment_filter_mask,
  )

  env_cfg = load_env_cfg(TASK_ID, play=True)
  shift = configure_v31_context(
    env_cfg,
    context=args.context,
    runtime_filter=training_runtime_filter,
    context_spec=environment_parameters(args.context),
    clearance_barrier_slope=args.clearance_barrier_slope,
    recovery_distance_m=RECOVERY_DISTANCE_M,
    filter_alpha=FILTER_ALPHA,
  )
  reward = configure_paper_dual_reward(
    env_cfg,
    args.candidate,
    runtime_filter_during_training=training_runtime_filter,
  )
  if args.candidate == "paper_stair_demo_scale":
    clearance = env_cfg.rewards["foot_clearance"]
    clearance.params = {
      **clearance.params,
      "reference_mode": "next_riser",
      "lookahead_distance": 0.60,
    }
    reward["clearance_reference"] = {
      "mode": "next_riser",
      "lookahead_distance_m": 0.60,
      "persists_after_cbf_deactivation": True,
    }
  shift["runtime_filter_fraction"] = training_filter_fraction
  reward["runtime_filter_fraction"] = training_filter_fraction
  deterministic_mean_teacher = None
  if args.deterministic_mean_teacher:
    from src.tasks.stairs_cbf.paper_teacher_v35 import (
      configure_v35_mean_teacher_telemetry,
    )

    deterministic_mean_teacher = configure_v35_mean_teacher_telemetry(
      env_cfg,
      runtime_filter_during_training=training_runtime_filter,
      failure_only=args.failure_only_mean_teacher,
      success_only=args.success_only_mean_teacher,
      failure_focused_actor=args.failure_focused_actor,
      distill_only_actor=args.distill_only_actor,
      success_local_kl_beta=args.success_local_kl_beta,
    )
    deterministic_mean_teacher["runtime_filter_fraction"] = (
      training_filter_fraction
    )
  height_curriculum = None
  if args.height_curriculum:
    height_curriculum = _configure_height_curriculum(
      env_cfg,
      shift,
      start_height=args.curriculum_start_height,
      num_rows=args.curriculum_rows,
    )
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK_ID)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  _configure_algorithm(agent_cfg, teacher_arms[0], preflight=False)
  agent_cfg.algorithm.learning_rate = float(args.actor_learning_rate)
  agent_cfg.algorithm.actor_learning_rate = float(args.actor_learning_rate)
  agent_cfg.algorithm.moving_kl_beta = float(args.moving_kl_beta)
  agent_cfg.algorithm.minimum_std = float(args.training_action_std)
  agent_cfg.algorithm.maximum_std = float(args.training_action_std)
  if args.deterministic_mean_teacher:
    agent_cfg.algorithm.class_name = (
      "src.tasks.stairs_cbf.paper_teacher_v35:PaperMeanTeacherV35PPO"
    )

  output_dir.mkdir(parents=True)
  started = time.monotonic()
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cfg = asdict(agent_cfg)
  if args.deterministic_mean_teacher:
    runner_cfg["algorithm"]["v35_failure_only_mean_teacher"] = (
      args.failure_only_mean_teacher
    )
    runner_cfg["algorithm"]["v35_success_only_mean_teacher"] = (
      args.success_only_mean_teacher
    )
    runner_cfg["algorithm"]["v35_failure_focused_actor"] = (
      args.failure_focused_actor
    )
    runner_cfg["algorithm"]["v35_distill_only_actor"] = (
      args.distill_only_actor
    )
    runner_cfg["algorithm"]["v35_success_local_kl_beta"] = (
      args.success_local_kl_beta
    )
  runner = CbfTeacherV30Runner(
    env, runner_cfg, log_dir=None, device=args.device
  )
  action_term = base_env.action_manager.get_term("joint_pos")

  def stage_deterministic_policy_mean(active_runner, _raw_actions) -> None:
    step = active_runner.alg.storage.step
    present = active_runner.alg.v30_reference_mean_present[step]
    if not bool(present.all()):
      missing = int((~present).sum())
      raise RuntimeError(f"v35 round-reference mean missing for {missing} envs")
    action_term.stage_counterfactual_policy_action(
      active_runner.alg.v30_reference_means[step]
    )

  before_env_step = (
    stage_deterministic_policy_mean
    if args.deterministic_mean_teacher
    else None
  )

  def stage_reached_top(active_runner, _dones, extras) -> None:
    extras["v35_reached_top"] = (
      active_runner.env.unwrapped.termination_manager.get_term("reached_top")
      .detach()
      .clone()
    )

  before_process_env_step = (
    stage_reached_top if args.deterministic_mean_teacher else None
  )
  records: list[dict[str, Any]] = []
  try:
    warm_start = runner.load_initial_checkpoint(
      str(checkpoint), map_location=args.device
    )
    initial_teacher_parameters = _set_teacher_arm(
      runner.alg,
      teacher_arms[0],
      a1_teacher_weight=args.a1_teacher_weight,
      a2_teacher_eta=args.a2_teacher_eta,
    )
    initial_hash = actor_state_sha256(actor_state(runner.alg.actor))
    _save_checkpoint(
      runner,
      output_dir / "round_00.pt",
      0,
      {
        "experiment": "paper_dual_v35",
        "candidate": args.candidate,
        "context": args.context,
        "teacher_arm": teacher_arms[0],
        "teacher_parameters": initial_teacher_parameters,
        "teacher_arms_by_round": teacher_arms,
        "height_curriculum": height_curriculum,
        "deterministic_mean_teacher": deterministic_mean_teacher,
        "success_only_mean_teacher": args.success_only_mean_teacher,
        "failure_focused_actor": args.failure_focused_actor,
        "distill_only_actor": args.distill_only_actor,
        "success_local_kl_beta": args.success_local_kl_beta,
        "training_runtime_filter": training_runtime_filter,
        "training_filter_fraction": training_filter_fraction,
        "training_action_std": args.training_action_std,
        "actor_learning_rate": args.actor_learning_rate,
        "moving_kl_beta": args.moving_kl_beta,
      },
    )
    for round_index in range(1, args.rounds + 1):
      round_teacher_arm = teacher_arms[round_index - 1]
      round_teacher_parameters = _set_teacher_arm(
        runner.alg,
        round_teacher_arm,
        a1_teacher_weight=args.a1_teacher_weight,
        a2_teacher_eta=args.a2_teacher_eta,
      )
      runner.alg.freeze_round_reference()
      start_hash = actor_state_sha256(actor_state(runner.alg.actor))
      round_filter_mask = rotating_environment_filter_mask(
        args.num_envs,
        training_filter_fraction,
        round_index,
        device=base_env.device,
      )
      action_term.set_runtime_filter_mask(round_filter_mask)
      round_started = time.monotonic()
      metrics = _collect_round(
        runner,
        before_env_step=before_env_step,
        before_process_env_step=before_process_env_step,
      )
      if height_curriculum is not None:
        metrics.update(
          _terrain_level_metrics(base_env, num_rows=args.curriculum_rows)
        )
      observed_filter_fraction = float(
        metrics["runtime_filter_enabled_fraction"]
      )
      expected_filter_fraction = float(round_filter_mask.float().mean())
      if not math.isclose(
        observed_filter_fraction,
        expected_filter_fraction,
        rel_tol=0.0,
        abs_tol=1.0e-8,
      ):
        raise RuntimeError("v35 runtime filter mask was not executed exactly")
      metrics["configured_runtime_filter_fraction"] = expected_filter_fraction
      metrics["configured_runtime_filter_count"] = int(round_filter_mask.sum())
      end_hash = actor_state_sha256(actor_state(runner.alg.actor))
      record = {
        "round": round_index,
        "status": "updated",
        "elapsed_seconds": time.monotonic() - round_started,
        "actor_sha256": end_hash,
        "round_start_actor_sha256": start_hash,
        "round_end_actor_sha256": end_hash,
        "rollout_actor_sha256": start_hash,
        "rollout_checkpoint_round": round_index - 1,
        "rollout_precedes_update": True,
        "runtime_filter_mask_rotation_round": round_index,
        "teacher_arm": round_teacher_arm,
        "teacher_parameters": round_teacher_parameters,
        "metrics": metrics,
      }
      records.append(record)
      _save_checkpoint(
        runner,
        output_dir / f"round_{round_index:02d}.pt",
        round_index,
        {
          "experiment": "paper_dual_v35",
          "candidate": args.candidate,
          "context": args.context,
          "teacher_arm": round_teacher_arm,
          "teacher_parameters": round_teacher_parameters,
          "teacher_arms_by_round": teacher_arms,
          "height_curriculum": height_curriculum,
          "deterministic_mean_teacher": deterministic_mean_teacher,
          "success_only_mean_teacher": args.success_only_mean_teacher,
          "failure_focused_actor": args.failure_focused_actor,
          "distill_only_actor": args.distill_only_actor,
          "success_local_kl_beta": args.success_local_kl_beta,
          "training_runtime_filter": training_runtime_filter,
          "training_filter_fraction": training_filter_fraction,
          "training_action_std": args.training_action_std,
          "actor_learning_rate": args.actor_learning_rate,
          "moving_kl_beta": args.moving_kl_beta,
        },
      )
      _atomic_json(output_dir / "round_metrics.json", records)
      _write_round_csv(output_dir / "round_metrics.csv", records)
      print(json.dumps(record, sort_keys=True), flush=True)
    final_checkpoint = output_dir / f"round_{args.rounds:02d}.pt"
    summary = {
      "schema_version": 1,
      "experiment": "paper_dual_v35",
      "git_commit": _git(repo, "rev-parse", "HEAD"),
      "context": args.context,
      "candidate": args.candidate,
      "teacher_arm": (
        args.teacher_arm if args.teacher_schedule == "fixed" else "staged"
      ),
      "teacher_schedule": args.teacher_schedule,
      "teacher_switch_after": (
        None if args.teacher_schedule == "fixed" else args.teacher_switch_after
      ),
      "a1_teacher_weight": args.a1_teacher_weight,
      "a2_teacher_eta": args.a2_teacher_eta,
      "teacher_arms_by_round": teacher_arms,
      "height_curriculum": height_curriculum,
      "deterministic_mean_teacher": deterministic_mean_teacher,
      "success_only_mean_teacher": args.success_only_mean_teacher,
      "failure_focused_actor": args.failure_focused_actor,
      "distill_only_actor": args.distill_only_actor,
      "success_local_kl_beta": args.success_local_kl_beta,
      "round_metric_actor_alignment": (
        "round_N_rollout_uses_round_N_minus_1_checkpoint"
      ),
      "training_runtime_filter": training_runtime_filter,
      "training_filter_fraction": training_filter_fraction,
      "training_action_std": args.training_action_std,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "seed": args.seed,
      "rounds": args.rounds,
      "num_envs": args.num_envs,
      "rollout_steps": args.rollout_steps,
      "shift": shift,
      "reward": reward,
      "warm_start": warm_start,
      "base_checkpoint": str(checkpoint),
      "base_checkpoint_sha256": checkpoint_sha256,
      "expected_base_checkpoint_sha256": expected_base_sha256,
      "base_checkpoint_role": (
        "common_online_refinement_base"
        if checkpoint_sha256 == BASE_CHECKPOINT_SHA256
        else "explicit_continuation"
      ),
      "initial_actor_sha256": initial_hash,
      "final_actor_sha256": actor_state_sha256(actor_state(runner.alg.actor)),
      "final_checkpoint": str(final_checkpoint),
      "final_checkpoint_sha256": file_sha256(final_checkpoint),
      "elapsed_seconds": time.monotonic() - started,
      "round_metrics": records,
    }
    _atomic_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
