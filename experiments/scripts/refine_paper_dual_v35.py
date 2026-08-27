"""Train one short paper-aligned dual CBF-RL reward candidate."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
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
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context", choices=("F1", "F2", "F3"), required=True)
  parser.add_argument(
    "--candidate",
    choices=("current", "raw_moderate", "raw_strong", "raw_demo"),
    required=True,
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
  parser.add_argument("--rounds", type=int, default=8)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=1024)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--device", default="cuda:0")
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


def _set_teacher_arm(algorithm, arm: str) -> dict[str, Any]:
  """Switch only the four frozen teacher fields at a round boundary."""
  parameters = arm_parameters(arm)
  algorithm.teacher_mode = parameters["teacher_mode"]
  algorithm.teacher_gate = parameters["teacher_gate"]
  algorithm.teacher_eta = parameters["teacher_eta"]
  algorithm.teacher_distillation_weight = parameters["teacher_weight"]
  return parameters


def main() -> None:
  args = _parse_args()
  if args.rounds < 1 or args.num_envs < 1 or args.rollout_steps < 1:
    raise ValueError("v35 rounds, environments, and rollout steps must be positive")
  teacher_arms = _teacher_arms_by_round(
    rounds=args.rounds,
    teacher_arm=args.teacher_arm,
    teacher_schedule=args.teacher_schedule,
    switch_after=args.teacher_switch_after,
  )
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  output_dir = args.output_dir.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  checkpoint_sha256 = file_sha256(checkpoint)
  if checkpoint_sha256 != BASE_CHECKPOINT_SHA256:
    raise RuntimeError(
      "v35 requires the common 838-D online-refinement base checkpoint: "
      f"{checkpoint_sha256} != {BASE_CHECKPOINT_SHA256}"
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
  reward = configure_paper_dual_reward(env_cfg, args.candidate)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK_ID)
  agent_cfg.seed = args.seed
  agent_cfg.num_steps_per_env = args.rollout_steps
  _configure_algorithm(agent_cfg, teacher_arms[0], preflight=False)

  output_dir.mkdir(parents=True)
  started = time.monotonic()
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner = CbfTeacherV30Runner(
    env, asdict(agent_cfg), log_dir=None, device=args.device
  )
  records: list[dict[str, Any]] = []
  try:
    warm_start = runner.load_initial_checkpoint(
      str(checkpoint), map_location=args.device
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
        "teacher_arms_by_round": teacher_arms,
      },
    )
    for round_index in range(1, args.rounds + 1):
      round_teacher_arm = teacher_arms[round_index - 1]
      round_teacher_parameters = _set_teacher_arm(
        runner.alg, round_teacher_arm
      )
      runner.alg.freeze_round_reference()
      start_hash = actor_state_sha256(actor_state(runner.alg.actor))
      round_started = time.monotonic()
      metrics = _collect_round(runner)
      end_hash = actor_state_sha256(actor_state(runner.alg.actor))
      record = {
        "round": round_index,
        "status": "updated",
        "elapsed_seconds": time.monotonic() - round_started,
        "actor_sha256": end_hash,
        "round_start_actor_sha256": start_hash,
        "round_end_actor_sha256": end_hash,
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
          "teacher_arms_by_round": teacher_arms,
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
      "teacher_arms_by_round": teacher_arms,
      "seed": args.seed,
      "rounds": args.rounds,
      "num_envs": args.num_envs,
      "rollout_steps": args.rollout_steps,
      "shift": shift,
      "reward": reward,
      "warm_start": warm_start,
      "base_checkpoint_sha256": checkpoint_sha256,
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
