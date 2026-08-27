"""Train one short paper-aligned dual CBF-RL reward candidate."""

from __future__ import annotations

import argparse
import hashlib
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
from paper_transactional_v73_math import (
  MINIMUM_ACTOR_LEARNING_RATE,
  REJECTED_CANDIDATE_LEARNING_RATE_SCALE,
  TARGET_MOVING_FORWARD_KL,
  adaptive_actor_learning_rate,
  rollout_candidate_decision,
)
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
      "paper_stair_sloped_exact",
      "paper_stair_sloped_unit_balanced",
      "paper_stair_sloped_mid_balanced",
      "paper_stair_sloped_proximity_balanced",
      "paper_stair_sloped_demo_scale",
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
    "--success-safe-action-imitation",
    action="store_true",
    help=(
      "On fully filtered rollouts, imitate executed safe actions only from "
      "complete reached-top episodes while retaining PPO and moving KL."
    ),
  )
  parser.add_argument(
    "--success-imitation-weight",
    type=float,
    default=0.5,
    help="Population-scaled Smooth-L1 coefficient for v88 success imitation.",
  )
  parser.add_argument(
    "--success-intervention-safe-mean-only",
    action="store_true",
    help=(
      "v89: restrict v88 imitation to successful deterministic-mean CBF "
      "interventions and target the same-state deterministic safe mean."
    ),
  )
  parser.add_argument(
    "--success-intervention-bounded-residual",
    action="store_true",
    help=(
      "v90: replace full safe-mean cloning with the configured bounded A2 "
      "fraction of the deterministic CBF correction."
    ),
  )
  parser.add_argument(
    "--success-residual-only-actor",
    action="store_true",
    help=(
      "v91: disable PPO and entropy actor gradients, retaining only the "
      "successful bounded residual teacher and moving reference KL."
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
  parser.add_argument(
    "--curriculum-freeze-target-after-round",
    type=int,
    default=None,
    help=(
      "At this rollout round, clamp every reset to the highest terrain row. "
      "This preserves early adaptive promotion but prevents easy-row retreat."
    ),
  )
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
    "--training-filter-schedule",
    choices=("fixed", "linear_to_off"),
    default="fixed",
    help=(
      "Keep one execution fraction, or anneal it linearly to a final "
      "unshielded rollout while retaining the counterfactual CBF reward."
    ),
  )
  parser.add_argument(
    "--filter-group-balanced-advantages",
    action="store_true",
    help=(
      "For a fixed mixed filter rollout, normalize PPO advantages separately "
      "over filtered and nominal environment groups."
    ),
  )
  parser.add_argument(
    "--split-filter-actor-objectives",
    action="store_true",
    help=(
      "Route nominal filter-off transitions to PPO actor gradients and "
      "filtered transitions to the deterministic-mean CBF teacher."
    ),
  )
  parser.add_argument(
    "--task-priority-gradient-surgery",
    action="store_true",
    help=(
      "When split objectives are enabled, project away only the CBF-teacher "
      "gradient component that conflicts with nominal PPO plus reference KL."
    ),
  )
  parser.add_argument(
    "--teacher-gradient-target-ratio",
    type=float,
    default=0.0,
    help=(
      "After task-priority projection, norm-balance the CBF-teacher gradient "
      "to this fraction of the deployment gradient; zero disables balancing."
    ),
  )
  parser.add_argument(
    "--full-batch-sgd-actor",
    action="store_true",
    help=(
      "Use one globally clipped full-batch SGD actor step per round instead "
      "of the historical eight Adam minibatch steps."
    ),
  )
  parser.add_argument(
    "--actor-gradient-accumulation-microbatches",
    type=int,
    default=1,
    help=(
      "Split one full-rollout actor gradient into this many equal backward "
      "chunks, then clip and execute one SGD step. Requires full-batch SGD."
    ),
  )
  parser.add_argument(
    "--transactional-rollout-acceptance",
    action="store_true",
    help=(
      "Evaluate each full-batch proposal on the next aligned rollout; restore "
      "the accepted actor/critic/optimizers after filter-off regression and "
      "shrink the next SGD step."
    ),
  )
  parser.add_argument(
    "--training-filter-end-fraction",
    type=float,
    default=0.0,
    help="Final execution fraction for linear_to_off (normally 0).",
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
  parser.add_argument(
    "--training-domain-randomization",
    choices=("off", "paper_static", "paper_full"),
    default="off",
    help=(
      "Restore native G1 training observation/parameter randomization; "
      "paper_full additionally restores interval pushes. Evaluation remains "
      "the fixed play environment."
    ),
  )
  parser.add_argument(
    "--training-domain-randomization-strength",
    type=float,
    default=1.0,
    help=(
      "Scale native perturbation ranges around their identity values; use a "
      "fraction below 1 when continuing a nominal pretrained gait."
    ),
  )
  parser.add_argument(
    "--training-domain-randomization-refresh",
    choices=("startup", "round"),
    default="startup",
    help=(
      "Keep one physical-parameter draw per environment, or refresh native "
      "startup DR terms at every rollout round."
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
  a2_teacher_weight: float | None = None,
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
    if a2_teacher_weight is not None:
      parameters["teacher_weight"] = float(a2_teacher_weight)
    parameters["name"] = (
      f"residual_eta_{a2_teacher_eta:g}_all_interventions"
      if a2_teacher_weight is None
      else (
        f"success_safe_action_imitation_weight_{a2_teacher_weight:g}"
      )
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
  minimum_level_schedule: tuple[int, ...],
) -> dict[str, Any]:
  """Turn one fixed uniform target into an ordered training curriculum."""
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

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
      func=_terrain_levels_vel_with_floor,
      params={"command_name": "twist"},
    )
  }
  return {
    "enabled": True,
    "start_height_m": start,
    "target_height_m": target,
    "num_rows": int(num_rows),
    "initial_level": 0,
    "promotion_rule": "terrain_levels_vel_with_monotone_floor",
    "minimum_level_schedule": list(minimum_level_schedule),
    "exact_per_level_cbf_geometry": True,
    "target_evaluation_remains_fixed": True,
  }


_HEIGHT_CURRICULUM_FLOOR_ATTRIBUTE = "_v35_height_curriculum_floor"


def _terrain_levels_vel_with_floor(env, env_ids, command_name: str) -> torch.Tensor:
  """Apply the native curriculum update, then enforce the current phase floor."""
  from src.tasks.velocity.mdp.curriculums import terrain_levels_vel

  terrain_levels_vel(env, env_ids, command_name)
  floor = int(getattr(env, _HEIGHT_CURRICULUM_FLOOR_ATTRIBUTE, 0))
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise RuntimeError("v60 terrain floor requires curriculum terrain origins")
  if not 0 <= floor < terrain.max_terrain_level:
    raise RuntimeError("v60 terrain floor lies outside generated terrain rows")
  levels = terrain.terrain_levels[env_ids]
  below_floor = levels < floor
  if bool(below_floor.any()):
    clamped_ids = env_ids[below_floor]
    terrain.terrain_levels[clamped_ids] = floor
    terrain.env_origins[clamped_ids] = terrain.terrain_origins[
      terrain.terrain_levels[clamped_ids], terrain.terrain_types[clamped_ids]
    ]
  return torch.mean(terrain.terrain_levels.float())


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


def _domain_randomization_state_sha256(base_env) -> str:
  """Fingerprint every per-world physical parameter and encoder-bias draw."""
  robot = base_env.scene["robot"]
  tensors = {
    "encoder_bias": robot.data.encoder_bias,
    "geom_friction": base_env.sim.model.geom_friction,
    "body_ipos": base_env.sim.model.body_ipos,
  }
  signature = hashlib.sha256()
  for name, tensor in tensors.items():
    signature.update(name.encode("utf-8"))
    signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return signature.hexdigest()


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
  if (
    args.candidate in {
      "paper_stair_sloped_exact",
      "paper_stair_sloped_unit_balanced",
      "paper_stair_sloped_mid_balanced",
      "paper_stair_sloped_proximity_balanced",
      "paper_stair_sloped_demo_scale",
    }
    and args.clearance_barrier_slope <= 0.0
  ):
    raise ValueError("sloped paper stair candidate requires a positive barrier slope")
  if not 0.0 < args.a1_teacher_weight <= 0.1:
    raise ValueError("v35 A1 teacher weight must be in (0, 0.1]")
  if (
    not math.isfinite(args.success_imitation_weight)
    or not 0.0 < args.success_imitation_weight <= 2.0
  ):
    raise ValueError("v88 success-imitation weight must lie in (0, 2]")
  if args.curriculum_rows < 2:
    raise ValueError("v35 curriculum rows must be at least two")
  if args.curriculum_freeze_target_after_round is not None:
    if not args.height_curriculum:
      raise ValueError("target-height freeze requires the height curriculum")
    if not 1 <= args.curriculum_freeze_target_after_round <= args.rounds:
      raise ValueError("target-height freeze round must lie within training rounds")
  if not 0.01 <= args.training_action_std <= 0.05:
    raise ValueError("v35 training action std must lie in [0.01, 0.05]")
  if args.full_batch_sgd_actor or args.success_safe_action_imitation:
    if not 1.0e-6 <= args.actor_learning_rate <= 1.0e-3:
      raise ValueError("v72 SGD learning rate must lie in [1e-6, 1e-3]")
  elif not 1.0e-7 <= args.actor_learning_rate <= 5.0e-6:
    raise ValueError("v35 actor learning rate must lie in [1e-7, 5e-6]")
  if not 0.0 <= args.moving_kl_beta <= 0.5:
    raise ValueError("v35 moving KL beta must lie in [0, 0.5]")
  if args.actor_gradient_accumulation_microbatches < 1:
    raise ValueError("actor gradient accumulation chunks must be positive")
  if (
    args.actor_gradient_accumulation_microbatches > 1
    and not (
      args.full_batch_sgd_actor or args.success_safe_action_imitation
    )
  ):
    raise ValueError(
      "actor gradient accumulation requires a single-step SGD actor"
    )
  if (
    args.training_domain_randomization_refresh == "round"
    and args.training_domain_randomization == "off"
  ):
    raise ValueError("round DR refresh requires training domain randomization")
  training_runtime_filter = args.training_runtime_filter == "on"
  training_filter_fraction = (
    1.0 if training_runtime_filter else 0.0
  ) if args.training_filter_fraction is None else float(
    args.training_filter_fraction
  )
  if not math.isfinite(training_filter_fraction) or not (
    0.0 <= training_filter_fraction <= 1.0
  ):
    raise ValueError("training filter fraction must lie in [0, 1]")
  if not training_runtime_filter and training_filter_fraction != 0.0:
    raise ValueError("disabled training filter requires fraction 0")
  if args.training_filter_schedule == "linear_to_off":
    if not training_runtime_filter:
      raise ValueError("filter annealing requires the CBF execution path enabled")
    if not 0.0 <= args.training_filter_end_fraction < training_filter_fraction:
      raise ValueError(
        "filter annealing end fraction must lie in [0, start fraction)"
      )
  if args.filter_group_balanced_advantages and (
    args.training_filter_schedule != "fixed"
    or not 0.0 < training_filter_fraction < 1.0
  ):
    raise ValueError(
      "filter-group-balanced advantages require a fixed mixed filter fraction"
    )
  teacher_arms = _teacher_arms_by_round(
    rounds=args.rounds,
    teacher_arm=args.teacher_arm,
    teacher_schedule=args.teacher_schedule,
    switch_after=args.teacher_switch_after,
  )
  if args.deterministic_mean_teacher and any(arm != "A2" for arm in teacher_arms):
    raise ValueError("v35 deterministic-mean teacher currently requires only A2")
  if args.success_safe_action_imitation:
    if not args.deterministic_mean_teacher or any(
      arm != "A2" for arm in teacher_arms
    ):
      raise ValueError("v88 success imitation requires deterministic A2 telemetry")
    if args.full_batch_sgd_actor:
      raise ValueError("v88 owns its SGD actor; do not also select v72")
    if (
      args.training_filter_schedule != "fixed"
      or training_filter_fraction != 1.0
      or args.filter_group_balanced_advantages
    ):
      raise ValueError("v88 success imitation requires fully filtered training")
    if any(
      (
        args.failure_only_mean_teacher,
        args.success_only_mean_teacher,
        args.failure_focused_actor,
        args.distill_only_actor,
        args.split_filter_actor_objectives,
        args.task_priority_gradient_surgery,
      )
    ) or args.success_local_kl_beta != 0.0:
      raise ValueError("v88 success imitation is mutually exclusive with v35 gates")
  if (
    args.success_intervention_safe_mean_only
    and not args.success_safe_action_imitation
  ):
    raise ValueError("v89 safe-mean restriction requires v88 success imitation")
  if (
    args.success_intervention_bounded_residual
    and not args.success_intervention_safe_mean_only
  ):
    raise ValueError("v90 bounded residual requires v89 success interventions")
  if (
    args.success_residual_only_actor
    and not args.success_intervention_bounded_residual
  ):
    raise ValueError("v91 residual-only actor requires v90 bounded residuals")
  if args.failure_only_mean_teacher and not args.deterministic_mean_teacher:
    raise ValueError("failure-only mean teacher requires deterministic mean labels")
  if args.failure_only_mean_teacher and args.training_runtime_filter != "off":
    raise ValueError("failure-only mean teacher requires unshielded training")
  if args.success_only_mean_teacher and not args.deterministic_mean_teacher:
    raise ValueError("success-only mean teacher requires deterministic mean labels")
  if args.failure_only_mean_teacher and args.success_only_mean_teacher:
    raise ValueError("mean-teacher outcome gates are mutually exclusive")
  if args.success_only_mean_teacher and (
    args.training_filter_schedule != "fixed"
    or training_filter_fraction != 1.0
  ):
    raise ValueError("success-only mean teacher requires fully shielded training")
  if args.failure_focused_actor and not args.failure_only_mean_teacher:
    raise ValueError(
      "failure-focused actor requires the failure-only mean teacher"
    )
  if args.distill_only_actor and not args.deterministic_mean_teacher:
    raise ValueError("distillation-only actor requires deterministic mean labels")
  if args.split_filter_actor_objectives:
    if (
      not args.deterministic_mean_teacher
      or args.distill_only_actor
      or args.failure_focused_actor
      or args.failure_only_mean_teacher
      or args.success_only_mean_teacher
    ):
      raise ValueError(
        "split filter actor objectives require ungated deterministic-mean A2"
      )
    if (
      args.training_filter_schedule != "fixed"
      or not 0.0 < training_filter_fraction < 1.0
      or not args.filter_group_balanced_advantages
    ):
      raise ValueError(
        "split filter actor objectives require fixed mixed execution and "
        "group-balanced advantages"
      )
  if (
    args.task_priority_gradient_surgery
    and not args.split_filter_actor_objectives
  ):
    raise ValueError(
      "task-priority gradient surgery requires split filter actor objectives"
    )
  if (
    not math.isfinite(args.teacher_gradient_target_ratio)
    or not 0.0 <= args.teacher_gradient_target_ratio <= 1.0
  ):
    raise ValueError("teacher gradient target ratio must lie in [0, 1]")
  if (
    args.teacher_gradient_target_ratio > 0.0
    and not args.task_priority_gradient_surgery
  ):
    raise ValueError(
      "teacher gradient norm balancing requires task-priority surgery"
    )
  if args.full_batch_sgd_actor:
    if (
      any(arm != "A0" for arm in teacher_arms)
      or args.deterministic_mean_teacher
      or args.split_filter_actor_objectives
      or args.task_priority_gradient_surgery
      or args.teacher_gradient_target_ratio != 0.0
    ):
      raise ValueError("v72 full-batch SGD requires teacher-free A0 training")
    mixed_group_balanced_execution = (
      args.training_filter_schedule == "fixed"
      and 0.0 < training_filter_fraction < 1.0
      and args.filter_group_balanced_advantages
    )
    paper_fully_filtered_execution = (
      args.training_filter_schedule == "fixed"
      and training_filter_fraction == 1.0
      and not args.filter_group_balanced_advantages
    )
    if not (
      mixed_group_balanced_execution or paper_fully_filtered_execution
    ):
      raise ValueError(
        "full-batch SGD requires either fixed mixed execution with "
        "group-balanced advantages or paper-style fully filtered execution"
      )
  if args.transactional_rollout_acceptance:
    if not (
      args.full_batch_sgd_actor or args.success_safe_action_imitation
    ):
      raise ValueError(
        "transactional acceptance requires a single-step SGD actor"
      )
    if args.rounds < 2:
      raise ValueError("v73 transactional acceptance requires at least 2 rounds")
    if args.training_domain_randomization != "off":
      raise ValueError(
        "v73 rollout acceptance requires a fixed physical training domain"
      )
  fully_filtered_full_batch_actor = bool(
    args.full_batch_sgd_actor
    and args.training_filter_schedule == "fixed"
    and training_filter_fraction == 1.0
  )
  fully_filtered_transactional_actor = bool(
    (args.full_batch_sgd_actor or args.success_safe_action_imitation)
    and args.training_filter_schedule == "fixed"
    and training_filter_fraction == 1.0
  )
  transactional_acceptance_group = (
    "filter_on" if fully_filtered_transactional_actor else "filter_off"
  )
  if not 0.0 <= args.success_local_kl_beta <= 4.0:
    raise ValueError("success-local KL beta must lie in [0, 4]")
  if args.success_local_kl_beta > 0.0:
    if (
      not args.deterministic_mean_teacher
      or args.failure_only_mean_teacher
      or args.success_only_mean_teacher
      or args.failure_focused_actor
    ):
      raise ValueError(
        "success-local KL requires all-intervention deterministic-mean labels"
      )
    shielded_distillation = (
      args.distill_only_actor
      and args.training_runtime_filter == "on"
      and args.training_filter_schedule == "fixed"
      and training_filter_fraction == 1.0
    )
    unshielded_ppo = (
      not args.distill_only_actor and args.training_runtime_filter == "off"
    )
    if not (shielded_distillation or unshielded_ppo):
      raise ValueError(
        "success-local KL requires either unshielded PPO or fully shielded "
        "distillation-only training"
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
  from src.tasks.stairs_cbf.paper_dual_v35 import (
    configure_paper_dual_reward,
    configure_paper_training_domain_randomization,
    normalize_filter_group_advantages,
  )
  from src.tasks.stairs_cbf.teacher_v30 import CbfTeacherV30Runner
  from src.tasks.stairs_cbf.teacher_v30_math import (
    linear_filter_fraction_schedule,
    rotating_environment_filter_mask,
    target_terrain_floor_schedule,
  )

  training_filter_fractions = (
    linear_filter_fraction_schedule(
      args.rounds,
      training_filter_fraction,
      args.training_filter_end_fraction,
    )
    if args.training_filter_schedule == "linear_to_off"
    else (training_filter_fraction,) * args.rounds
  )
  height_floor_schedule = target_terrain_floor_schedule(
    args.rounds,
    args.curriculum_rows,
    args.curriculum_freeze_target_after_round,
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
  training_domain_randomization = (
    configure_paper_training_domain_randomization(
      env_cfg,
      args.training_domain_randomization,
      strength=args.training_domain_randomization_strength,
    )
  )
  training_domain_randomization["refresh_mode"] = (
    args.training_domain_randomization_refresh
  )
  physical_draw_sets = (
    args.rounds
    if training_domain_randomization["enabled"]
    and args.training_domain_randomization_refresh == "round"
    else int(training_domain_randomization["enabled"])
  )
  training_domain_randomization["physical_parameter_draw_sets"] = (
    physical_draw_sets
  )
  training_domain_randomization["physical_parameter_draw_count"] = (
    physical_draw_sets * args.num_envs
  )
  if training_domain_randomization["enabled"]:
    evaluation_contract = shift.pop("fixed_deployment_environment", None)
    if evaluation_contract is not None:
      shift["fixed_evaluation_environment"] = evaluation_contract
    shift["friction_changed"] = True
    shift["actor_observation_corruption_changed"] = True
    shift["physical_parameter_randomization_changed"] = True
  shift["training_domain_randomization"] = training_domain_randomization
  if args.candidate in {
    "paper_stair_exact",
    "paper_stair_demo_scale",
    "paper_stair_sloped_exact",
    "paper_stair_sloped_unit_balanced",
    "paper_stair_sloped_mid_balanced",
    "paper_stair_sloped_proximity_balanced",
    "paper_stair_sloped_demo_scale",
  }:
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
    reward["clearance_barrier_geometry"] = (
      "paper_horizontal"
      if args.clearance_barrier_slope == 0.0
      else "task_compatible_sloped"
    )
  filter_schedule = {
    "name": args.training_filter_schedule,
    "fractions_by_round": list(training_filter_fractions),
    "start_fraction": training_filter_fractions[0],
    "end_fraction": training_filter_fractions[-1],
    "counterfactual_cbf_reward_retained_when_unshielded": True,
  }
  shift["runtime_filter_fraction"] = training_filter_fraction
  shift["runtime_filter_schedule"] = filter_schedule
  reward["runtime_filter_fraction"] = training_filter_fraction
  reward["runtime_filter_schedule"] = filter_schedule
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
      split_filter_actor_objectives=args.split_filter_actor_objectives,
      task_priority_gradient_surgery=(
        args.task_priority_gradient_surgery
      ),
      teacher_gradient_target_ratio=(
        args.teacher_gradient_target_ratio
      ),
    )
    deterministic_mean_teacher["runtime_filter_fraction"] = (
      training_filter_fraction
    )
    deterministic_mean_teacher["success_safe_action_imitation"] = (
      args.success_safe_action_imitation
    )
    deterministic_mean_teacher["success_imitation_weight"] = (
      args.success_imitation_weight
      if args.success_safe_action_imitation
      else 0.0
    )
    deterministic_mean_teacher["success_intervention_safe_mean_only"] = (
      args.success_intervention_safe_mean_only
    )
    deterministic_mean_teacher["success_intervention_bounded_residual"] = (
      args.success_intervention_bounded_residual
    )
    deterministic_mean_teacher["success_residual_only_actor"] = (
      args.success_residual_only_actor
    )
  height_curriculum = None
  if args.height_curriculum:
    height_curriculum = _configure_height_curriculum(
      env_cfg,
      shift,
      start_height=args.curriculum_start_height,
      num_rows=args.curriculum_rows,
      minimum_level_schedule=height_floor_schedule,
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
  if args.success_safe_action_imitation:
    agent_cfg.algorithm.class_name = (
      "src.tasks.stairs_cbf.paper_success_residual_only_v91:"
      "PaperSuccessResidualOnlyV91PPO"
      if args.success_residual_only_actor
      else (
        "src.tasks.stairs_cbf.paper_success_residual_v90:"
        "PaperSuccessResidualV90PPO"
        if args.success_intervention_bounded_residual
        else (
          "src.tasks.stairs_cbf.paper_success_intervention_v89:"
          "PaperSuccessInterventionV89PPO"
          if args.success_intervention_safe_mean_only
          else (
            "src.tasks.stairs_cbf.paper_success_imitation_v88:"
            "PaperSuccessImitationV88PPO"
          )
        )
      )
    )
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = (
      args.actor_gradient_accumulation_microbatches
    )
  elif args.full_batch_sgd_actor:
    if args.actor_gradient_accumulation_microbatches > 1:
      agent_cfg.algorithm.class_name = (
        "src.tasks.stairs_cbf.paper_accumulated_v82:PaperAccumulatedV82PPO"
      )
    else:
      agent_cfg.algorithm.class_name = (
        "src.tasks.stairs_cbf.paper_full_filter_v75:PaperFullFilterV75PPO"
        if fully_filtered_full_batch_actor
        else "src.tasks.stairs_cbf.paper_full_batch_v72:PaperFullBatchV72PPO"
      )
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = (
      args.actor_gradient_accumulation_microbatches
    )
  elif args.deterministic_mean_teacher:
    agent_cfg.algorithm.class_name = (
      "src.tasks.stairs_cbf.paper_teacher_v35:PaperMeanTeacherV35PPO"
    )

  output_dir.mkdir(parents=True)
  started = time.monotonic()
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  setattr(base_env, _HEIGHT_CURRICULUM_FLOOR_ATTRIBUTE, 0)
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
    runner_cfg["algorithm"]["v35_split_filter_actor_objectives"] = (
      args.split_filter_actor_objectives
    )
    runner_cfg["algorithm"]["v35_task_priority_gradient_surgery"] = (
      args.task_priority_gradient_surgery
    )
    runner_cfg["algorithm"]["v35_teacher_gradient_target_ratio"] = (
      args.teacher_gradient_target_ratio
    )
    if args.success_safe_action_imitation:
      runner_cfg["algorithm"]["v88_success_imitation_weight"] = (
        args.success_imitation_weight
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
  observed_dr_state_hashes: set[str] = set()
  accepted_transaction = None
  accepted_actor_sha256: str | None = None
  accepted_success_count: int | None = None
  accepted_episode_count: int | None = None
  accepted_success_rate: float | None = None
  accepted_checkpoint: Path | None = None
  accepted_rollout_round: int | None = None
  transactional_rollback_count = 0

  def set_actor_learning_rate(learning_rate: float) -> None:
    runner.alg.actor_learning_rate = float(learning_rate)
    runner.alg.learning_rate = float(learning_rate)
    for parameter_group in runner.alg.actor_optimizer.param_groups:
      parameter_group["lr"] = float(learning_rate)

  try:
    warm_start = runner.load_initial_checkpoint(
      str(checkpoint), map_location=args.device
    )
    initial_teacher_parameters = _set_teacher_arm(
      runner.alg,
      teacher_arms[0],
      a1_teacher_weight=args.a1_teacher_weight,
      a2_teacher_eta=args.a2_teacher_eta,
      a2_teacher_weight=(
        args.success_imitation_weight
        if args.success_safe_action_imitation
        else None
      ),
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
        "success_safe_action_imitation": args.success_safe_action_imitation,
        "success_intervention_safe_mean_only": (
          args.success_intervention_safe_mean_only
        ),
        "success_intervention_bounded_residual": (
          args.success_intervention_bounded_residual
        ),
        "success_residual_only_actor": args.success_residual_only_actor,
        "success_imitation_weight": (
          args.success_imitation_weight
          if args.success_safe_action_imitation
          else 0.0
        ),
        "failure_focused_actor": args.failure_focused_actor,
        "distill_only_actor": args.distill_only_actor,
        "success_local_kl_beta": args.success_local_kl_beta,
        "training_runtime_filter": training_runtime_filter,
        "training_filter_fraction": training_filter_fraction,
        "training_filter_schedule": filter_schedule,
        "filter_group_balanced_advantages": (
          args.filter_group_balanced_advantages
        ),
        "split_filter_actor_objectives": (
          args.split_filter_actor_objectives
        ),
        "task_priority_gradient_surgery": (
          args.task_priority_gradient_surgery
        ),
        "teacher_gradient_target_ratio": (
          args.teacher_gradient_target_ratio
        ),
        "full_batch_sgd_actor": args.full_batch_sgd_actor,
        "actor_gradient_accumulation_microbatches": (
          args.actor_gradient_accumulation_microbatches
        ),
        "fully_filtered_full_batch_actor": fully_filtered_full_batch_actor,
        "fully_filtered_transactional_actor": (
          fully_filtered_transactional_actor
        ),
        "transactional_rollout_acceptance": (
          args.transactional_rollout_acceptance
        ),
        "transactional_acceptance_group": (
          transactional_acceptance_group
          if args.transactional_rollout_acceptance
          else None
        ),
        "training_action_std": args.training_action_std,
        "actor_learning_rate": args.actor_learning_rate,
        "moving_kl_beta": args.moving_kl_beta,
        "training_domain_randomization": training_domain_randomization,
      },
    )
    for round_index in range(1, args.rounds + 1):
      dr_state_sha256 = None
      if training_domain_randomization["enabled"]:
        if args.training_domain_randomization_refresh == "round":
          base_env.event_manager.apply(mode="startup")
        dr_state_sha256 = _domain_randomization_state_sha256(base_env)
        if (
          args.training_domain_randomization_refresh == "round"
          and dr_state_sha256 in observed_dr_state_hashes
        ):
          raise RuntimeError("v62 round DR refresh repeated a parameter state")
        observed_dr_state_hashes.add(dr_state_sha256)
      round_terrain_floor = height_floor_schedule[round_index - 1]
      setattr(
        base_env,
        _HEIGHT_CURRICULUM_FLOOR_ATTRIBUTE,
        round_terrain_floor,
      )
      round_teacher_arm = teacher_arms[round_index - 1]
      round_teacher_parameters = _set_teacher_arm(
        runner.alg,
        round_teacher_arm,
        a1_teacher_weight=args.a1_teacher_weight,
        a2_teacher_eta=args.a2_teacher_eta,
        a2_teacher_weight=(
          args.success_imitation_weight
          if args.success_safe_action_imitation
          else None
        ),
      )
      runner.alg.freeze_round_reference()
      transaction = (
        runner.snapshot_proximal_state()
        if args.transactional_rollout_acceptance
        else None
      )
      start_hash = actor_state_sha256(actor_state(runner.alg.actor))
      candidate_checkpoint = output_dir / f"round_{round_index - 1:02d}.pt"
      candidate_checkpoint_sha256 = file_sha256(candidate_checkpoint)
      actor_learning_rate_used = float(
        runner.alg.actor_optimizer.param_groups[0]["lr"]
      )
      round_filter_fraction = training_filter_fractions[round_index - 1]
      round_filter_mask = rotating_environment_filter_mask(
        args.num_envs,
        round_filter_fraction,
        round_index,
        device=base_env.device,
      )
      action_term.set_runtime_filter_mask(round_filter_mask)
      if args.split_filter_actor_objectives:
        runner.alg.set_v35_filter_execution_environment_mask(
          round_filter_mask
        )
      after_compute_returns = None
      if args.filter_group_balanced_advantages:
        def after_compute_returns(active_runner):
          advantages = active_runner.alg.storage.advantages.squeeze(-1)
          balanced, advantage_metrics = normalize_filter_group_advantages(
            advantages, round_filter_mask.to(advantages.device)
          )
          active_runner.alg.storage.advantages.copy_(balanced.unsqueeze(-1))
          return advantage_metrics

      round_started = time.monotonic()
      metrics = _collect_round(
        runner,
        before_env_step=before_env_step,
        before_process_env_step=before_process_env_step,
        after_compute_returns=after_compute_returns,
        rollout_group_masks={
          "filter_on": round_filter_mask,
          "filter_off": ~round_filter_mask,
        },
      )
      metrics["training_domain_randomization_state_sha256"] = dr_state_sha256
      metrics["training_domain_randomization_refresh_mode"] = (
        args.training_domain_randomization_refresh
      )
      metrics["training_domain_randomization_distinct_state_count"] = len(
        observed_dr_state_hashes
      )
      for count_name in ("episode", "success", "fall", "timeout"):
        grouped_count = (
          metrics[f"rollout_filter_on_{count_name}_count"]
          + metrics[f"rollout_filter_off_{count_name}_count"]
        )
        total_count = metrics[f"rollout_{count_name}_count"]
        if grouped_count != total_count:
          raise RuntimeError(
            f"filter subgroup {count_name} count mismatch: "
            f"groups={grouped_count}, total={total_count}"
          )
      if height_curriculum is not None:
        metrics.update(
          _terrain_level_metrics(base_env, num_rows=args.curriculum_rows)
        )
        metrics["terrain_level_floor"] = round_terrain_floor
        if metrics["terrain_level_min"] < round_terrain_floor:
          raise RuntimeError("v60 terrain level escaped its configured floor")
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
      post_update_hash = actor_state_sha256(actor_state(runner.alg.actor))
      candidate_decision = None
      transactional_rollback_applied = False
      next_actor_learning_rate = actor_learning_rate_used
      if args.transactional_rollout_acceptance:
        candidate_decision = rollout_candidate_decision(
          actor_sha256=start_hash,
          success_count=int(
            metrics[
              f"rollout_{transactional_acceptance_group}_success_count"
            ]
          ),
          episode_count=int(
            metrics[
              f"rollout_{transactional_acceptance_group}_episode_count"
            ]
          ),
          accepted_actor_sha256=accepted_actor_sha256,
          accepted_success_count=accepted_success_count,
          accepted_episode_count=accepted_episode_count,
        )
        if candidate_decision["replace_anchor"]:
          assert transaction is not None
          accepted_transaction = transaction
          accepted_actor_sha256 = start_hash
          accepted_checkpoint = candidate_checkpoint
          accepted_rollout_round = round_index
        if candidate_decision["accepted"]:
          accepted_success_count = int(
            candidate_decision["anchor_success_count_after"]
          )
          accepted_episode_count = int(
            candidate_decision["anchor_episode_count_after"]
          )
          accepted_success_rate = float(
            candidate_decision["anchor_success_rate_after"]
          )
        if not candidate_decision["accepted"]:
          if accepted_transaction is None:
            raise RuntimeError("v73 rejected a candidate without an anchor")
          runner.restore_proximal_state(accepted_transaction)
          transactional_rollback_applied = True
          transactional_rollback_count += 1
        next_actor_learning_rate = adaptive_actor_learning_rate(
          actor_learning_rate_used,
          float(metrics["moving_forward_kl"]),
          rejected=transactional_rollback_applied,
        )
        set_actor_learning_rate(next_actor_learning_rate)
        metrics.update(
          {
            "transactional_rollout_acceptance_enabled": True,
            "transactional_acceptance_group": (
              transactional_acceptance_group
            ),
            "transactional_candidate_accepted": bool(
              candidate_decision["accepted"]
            ),
            "transactional_candidate_replace_anchor": bool(
              candidate_decision["replace_anchor"]
            ),
            "transactional_candidate_same_actor_retry": bool(
              candidate_decision["same_actor_retry"]
            ),
            "transactional_candidate_reason": candidate_decision["reason"],
            "transactional_candidate_improvement_percentage_points": (
              candidate_decision["improvement_percentage_points"]
            ),
            "transactional_rollback_applied": (
              transactional_rollback_applied
            ),
            "transactional_actor_learning_rate_used": (
              actor_learning_rate_used
            ),
            "transactional_next_actor_learning_rate": (
              next_actor_learning_rate
            ),
            "transactional_target_moving_forward_kl": (
              TARGET_MOVING_FORWARD_KL
            ),
            "transactional_accepted_actor_sha256": accepted_actor_sha256,
            "transactional_accepted_success_count": accepted_success_count,
            "transactional_accepted_episode_count": accepted_episode_count,
            "transactional_accepted_success_rate": accepted_success_rate,
            "transactional_accepted_filter_off_success_rate": (
              accepted_success_rate
              if transactional_acceptance_group == "filter_off"
              else None
            ),
            "transactional_accepted_filter_on_success_rate": (
              accepted_success_rate
              if transactional_acceptance_group == "filter_on"
              else None
            ),
          }
        )
      end_hash = actor_state_sha256(actor_state(runner.alg.actor))
      record = {
        "round": round_index,
        "status": "updated",
        "elapsed_seconds": time.monotonic() - round_started,
        "actor_sha256": end_hash,
        "round_start_actor_sha256": start_hash,
        "post_update_actor_sha256": post_update_hash,
        "round_end_actor_sha256": end_hash,
        "rollout_actor_sha256": start_hash,
        "rollout_checkpoint_round": round_index - 1,
        "rollout_checkpoint": str(candidate_checkpoint),
        "rollout_checkpoint_sha256": candidate_checkpoint_sha256,
        "rollout_precedes_update": True,
        "runtime_filter_mask_rotation_round": round_index,
        "scheduled_runtime_filter_fraction": round_filter_fraction,
        "teacher_arm": round_teacher_arm,
        "teacher_parameters": round_teacher_parameters,
        "transactional_candidate_decision": candidate_decision,
        "transactional_rollback_applied": transactional_rollback_applied,
        "actor_learning_rate_used": actor_learning_rate_used,
        "next_actor_learning_rate": next_actor_learning_rate,
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
          "success_safe_action_imitation": (
            args.success_safe_action_imitation
          ),
          "success_intervention_safe_mean_only": (
            args.success_intervention_safe_mean_only
          ),
          "success_intervention_bounded_residual": (
            args.success_intervention_bounded_residual
          ),
          "success_residual_only_actor": args.success_residual_only_actor,
          "success_imitation_weight": (
            args.success_imitation_weight
            if args.success_safe_action_imitation
            else 0.0
          ),
          "failure_focused_actor": args.failure_focused_actor,
          "distill_only_actor": args.distill_only_actor,
          "success_local_kl_beta": args.success_local_kl_beta,
          "training_runtime_filter": training_runtime_filter,
          "training_filter_fraction": training_filter_fraction,
          "training_filter_schedule": filter_schedule,
          "filter_group_balanced_advantages": (
            args.filter_group_balanced_advantages
          ),
          "split_filter_actor_objectives": (
            args.split_filter_actor_objectives
          ),
          "task_priority_gradient_surgery": (
            args.task_priority_gradient_surgery
          ),
          "teacher_gradient_target_ratio": (
            args.teacher_gradient_target_ratio
          ),
          "full_batch_sgd_actor": args.full_batch_sgd_actor,
          "actor_gradient_accumulation_microbatches": (
            args.actor_gradient_accumulation_microbatches
          ),
          "fully_filtered_full_batch_actor": fully_filtered_full_batch_actor,
          "fully_filtered_transactional_actor": (
            fully_filtered_transactional_actor
          ),
          "transactional_rollout_acceptance": (
            args.transactional_rollout_acceptance
          ),
          "transactional_acceptance_group": (
            transactional_acceptance_group
            if args.transactional_rollout_acceptance
            else None
          ),
          "training_action_std": args.training_action_std,
          "actor_learning_rate": args.actor_learning_rate,
          "moving_kl_beta": args.moving_kl_beta,
          "training_domain_randomization": training_domain_randomization,
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
      "success_safe_action_imitation": args.success_safe_action_imitation,
      "success_intervention_safe_mean_only": (
        args.success_intervention_safe_mean_only
      ),
      "success_intervention_bounded_residual": (
        args.success_intervention_bounded_residual
      ),
      "success_residual_only_actor": args.success_residual_only_actor,
      "success_imitation_weight": (
        args.success_imitation_weight
        if args.success_safe_action_imitation
        else 0.0
      ),
      "failure_focused_actor": args.failure_focused_actor,
      "distill_only_actor": args.distill_only_actor,
      "success_local_kl_beta": args.success_local_kl_beta,
      "round_metric_actor_alignment": (
        "round_N_rollout_uses_round_N_minus_1_checkpoint"
      ),
      "training_runtime_filter": training_runtime_filter,
      "training_filter_fraction": training_filter_fraction,
      "training_filter_schedule": filter_schedule,
      "filter_group_balanced_advantages": (
        args.filter_group_balanced_advantages
      ),
      "split_filter_actor_objectives": (
        args.split_filter_actor_objectives
      ),
      "task_priority_gradient_surgery": (
        args.task_priority_gradient_surgery
      ),
      "teacher_gradient_target_ratio": (
        args.teacher_gradient_target_ratio
      ),
      "full_batch_sgd_actor": args.full_batch_sgd_actor,
      "actor_gradient_accumulation_microbatches": (
        args.actor_gradient_accumulation_microbatches
      ),
      "fully_filtered_full_batch_actor": fully_filtered_full_batch_actor,
      "fully_filtered_transactional_actor": fully_filtered_transactional_actor,
      "transactional_rollout_acceptance": (
        args.transactional_rollout_acceptance
      ),
      "transactional_acceptance_group": (
        transactional_acceptance_group
        if args.transactional_rollout_acceptance
        else None
      ),
      "transactional_target_moving_forward_kl": (
        TARGET_MOVING_FORWARD_KL
        if args.transactional_rollout_acceptance
        else None
      ),
      "transactional_minimum_actor_learning_rate": (
        MINIMUM_ACTOR_LEARNING_RATE
        if args.transactional_rollout_acceptance
        else None
      ),
      "transactional_rejected_candidate_learning_rate_scale": (
        REJECTED_CANDIDATE_LEARNING_RATE_SCALE
        if args.transactional_rollout_acceptance
        else None
      ),
      "transactional_rollback_count": transactional_rollback_count,
      "selected_checkpoint": (
        str(accepted_checkpoint)
        if accepted_checkpoint is not None
        else None
      ),
      "selected_checkpoint_sha256": (
        file_sha256(accepted_checkpoint)
        if accepted_checkpoint is not None
        else None
      ),
      "selected_actor_sha256": accepted_actor_sha256,
      "selected_success_count": accepted_success_count,
      "selected_episode_count": accepted_episode_count,
      "selected_success_rate": accepted_success_rate,
      "selected_filter_off_success_rate": (
        accepted_success_rate
        if transactional_acceptance_group == "filter_off"
        and args.transactional_rollout_acceptance
        else None
      ),
      "selected_filter_on_success_rate": (
        accepted_success_rate
        if transactional_acceptance_group == "filter_on"
        and args.transactional_rollout_acceptance
        else None
      ),
      "selected_rollout_round": accepted_rollout_round,
      "training_action_std": args.training_action_std,
      "actor_learning_rate": args.actor_learning_rate,
      "moving_kl_beta": args.moving_kl_beta,
      "training_domain_randomization": training_domain_randomization,
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
