"""Run the frozen F1/F2/F3 filter-free adaptation matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tasks.stairs_cbf.paper_filter_free_ablation import (  # noqa: E402
  ADAPTATION_ARMS,
  CONTEXTS,
  INITIAL_ACTOR_LEARNING_RATE,
  NUM_ENVS,
  ROLLOUT_STEPS,
  ROUNDS,
  TRAINING_ACTION_STD,
  TRAINING_SEEDS,
  V139_SELECTED_CHECKPOINT_SHA256,
  arm_variables,
)

RUN_ORDER = (
  "dual_safe_ft",
  "filter_only_ft",
  "reward_only_ft",
  "nominal_ft",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, default=REPO_ROOT)
  parser.add_argument("--python", type=Path, default=Path(sys.executable))
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=CONTEXTS)
  parser.add_argument(
    "--arms", nargs="+", choices=ADAPTATION_ARMS, default=RUN_ORDER
  )
  parser.add_argument(
    "--seeds", nargs="+", type=int, choices=TRAINING_SEEDS, default=TRAINING_SEEDS
  )
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _completed_summary(
  output_dir: Path,
  *,
  context: str,
  arm: str,
  seed: int,
  git_commit: str,
) -> dict[str, Any] | None:
  summary_path = output_dir / "training_summary.json"
  if not summary_path.is_file():
    return None
  summary = json.loads(summary_path.read_text())
  metadata = summary.get("paper_filter_free_ablation_training") or {}
  expected = {
    "git_commit": git_commit,
    "context": context,
    "seed": seed,
    "rounds": ROUNDS,
    "num_envs": NUM_ENVS,
    "rollout_steps": ROLLOUT_STEPS,
    "base_checkpoint_sha256": V139_SELECTED_CHECKPOINT_SHA256,
  }
  mismatches = {
    key: (summary.get(key), value)
    for key, value in expected.items()
    if summary.get(key) != value
  }
  if metadata.get("arm") != arm:
    mismatches["arm"] = (metadata.get("arm"), arm)
  final_checkpoint = output_dir / f"round_{ROUNDS:02d}.pt"
  if mismatches or not final_checkpoint.is_file():
    raise RuntimeError(
      f"existing formal run is not reusable: {output_dir}, mismatches={mismatches}"
    )
  return summary


def _command(
  args: argparse.Namespace,
  *,
  context: str,
  arm: str,
  seed: int,
  output_dir: Path,
) -> list[str]:
  factors = arm_variables(arm)
  filter_enabled = factors["runtime_filter_during_adaptation"]
  return [
    str(args.python.resolve()),
    str(args.repo / "experiments/scripts/refine_paper_dual_v35.py"),
    "--repo",
    str(args.repo),
    "--base-checkpoint",
    str(args.base_checkpoint),
    "--expected-base-sha256",
    V139_SELECTED_CHECKPOINT_SHA256,
    "--output-dir",
    str(output_dir),
    "--context",
    context,
    "--candidate",
    "paper_stair_sloped_unit_balanced",
    "--clearance-barrier-slope",
    "0.8",
    "--cbf-mode",
    "current",
    "--actor-observation-interface",
    "original-405",
    "--teacher-arm",
    "A0",
    "--rounds",
    str(ROUNDS),
    "--num-envs",
    str(NUM_ENVS),
    "--rollout-steps",
    str(ROLLOUT_STEPS),
    "--seed",
    str(seed),
    "--device",
    args.device,
    "--training-runtime-filter",
    "on" if filter_enabled else "off",
    "--training-filter-fraction",
    "1.0" if filter_enabled else "0.0",
    "--training-filter-schedule",
    "fixed",
    "--training-action-std",
    str(TRAINING_ACTION_STD),
    "--actor-learning-rate",
    str(INITIAL_ACTOR_LEARNING_RATE),
    "--moving-kl-beta",
    "0.0",
    "--training-domain-randomization",
    "off",
    "--actor-gradient-accumulation-microbatches",
    "1",
    "--paper-filter-free-ablation-training",
    "--filter-free-ablation-arm",
    arm,
  ]


def main() -> None:
  args = _parse_args()
  args.repo = args.repo.resolve()
  args.base_checkpoint = args.base_checkpoint.resolve()
  args.output_root = args.output_root.resolve()
  if not args.base_checkpoint.is_file():
    raise FileNotFoundError(args.base_checkpoint)
  if not args.python.is_file():
    raise FileNotFoundError(args.python)
  if _git(args.repo, "status", "--porcelain"):
    raise RuntimeError("formal filter-free matrix requires a clean worktree")

  git_commit = _git(args.repo, "rev-parse", "HEAD")
  jobs = [
    (context, arm, seed)
    for context in args.contexts
    for arm in args.arms
    for seed in args.seeds
  ]
  manifest = {
    "schema_version": 1,
    "experiment": "paper_filter_free_ablation_training_matrix",
    "git_commit": git_commit,
    "base_checkpoint": str(args.base_checkpoint),
    "base_checkpoint_sha256": V139_SELECTED_CHECKPOINT_SHA256,
    "contexts": list(args.contexts),
    "arms": list(args.arms),
    "seeds": list(args.seeds),
    "rounds": ROUNDS,
    "num_envs": NUM_ENVS,
    "rollout_steps": ROLLOUT_STEPS,
    "run_order": [
      {"context": context, "arm": arm, "seed": seed}
      for context, arm, seed in jobs
    ],
    "started_unix": time.time(),
  }
  if args.dry_run:
    manifest["commands"] = [
      _command(
        args,
        context=context,
        arm=arm,
        seed=seed,
        output_dir=args.output_root / context / arm / f"seed_{seed}",
      )
      for context, arm, seed in jobs
    ]
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return

  args.output_root.mkdir(parents=True, exist_ok=True)
  log_root = args.output_root / "_logs"
  log_root.mkdir(parents=True, exist_ok=True)
  _atomic_json(args.output_root / "training_manifest.json", manifest)
  durations: list[float] = []
  completed = 0
  skipped = 0
  for job_index, (context, arm, seed) in enumerate(jobs, start=1):
    output_dir = args.output_root / context / arm / f"seed_{seed}"
    summary = _completed_summary(
      output_dir,
      context=context,
      arm=arm,
      seed=seed,
      git_commit=git_commit,
    )
    if summary is not None:
      completed += 1
      skipped += 1
      continue
    if output_dir.exists():
      raise RuntimeError(f"incomplete output directory requires review: {output_dir}")

    command = _command(
      args,
      context=context,
      arm=arm,
      seed=seed,
      output_dir=output_dir,
    )
    log_path = log_root / f"{context}_{arm}_seed_{seed}.log"
    progress = {
      **manifest,
      "status": "running",
      "job_index": job_index,
      "job_count": len(jobs),
      "current_job": {"context": context, "arm": arm, "seed": seed},
      "completed_jobs": completed,
      "skipped_jobs": skipped,
      "estimated_remaining_seconds": (
        None if not durations else (len(jobs) - completed) * sum(durations) / len(durations)
      ),
      "updated_unix": time.time(),
    }
    _atomic_json(args.output_root / "training_progress.json", progress)
    print(json.dumps(progress, sort_keys=True), flush=True)
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(
      {
        "PYTHONPATH": str(args.repo),
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
      }
    )
    with log_path.open("w") as log_file:
      subprocess.run(
        command,
        cwd=args.repo,
        env=environment,
        check=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
      )
    durations.append(time.monotonic() - started)
    completed += 1

  finished = {
    **manifest,
    "status": "complete",
    "completed_jobs": completed,
    "skipped_jobs": skipped,
    "job_count": len(jobs),
    "mean_job_seconds": None if not durations else sum(durations) / len(durations),
    "finished_unix": time.time(),
  }
  _atomic_json(args.output_root / "training_progress.json", finished)
  print(json.dumps(finished, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
