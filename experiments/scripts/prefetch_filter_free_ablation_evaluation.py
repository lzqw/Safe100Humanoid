"""Precompute a disjoint suffix of formal paired-evaluation conditions.

The primary runner remains the sole writer of the formal manifest, progress,
and paired result table.  This helper only writes normal evaluator outputs for
a prospectively disjoint suffix of checkpoint specs; the primary runner later
validates and reuses those outputs exactly as if they had been produced in its
own sequential loop.
"""

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

from experiments.scripts.run_filter_free_ablation_evaluation import (  # noqa: E402
  FILTER_CONDITIONS,
  PAIRED_EVALUATION_EPISODES,
  _condition_dir,
  _sha256,
  _training_checkpoint_specs,
  _valid_existing,
)
from src.tasks.stairs_cbf.paper_filter_free_ablation import (  # noqa: E402
  EVALUATION_SEED,
  V139_SELECTED_CHECKPOINT_SHA256,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, default=REPO_ROOT)
  parser.add_argument("--python", type=Path, default=Path(sys.executable))
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
  parser.add_argument(
    "--start-spec-index",
    type=int,
    required=True,
    help="Zero-based inclusive checkpoint-spec suffix boundary.",
  )
  parser.add_argument(
    "--reverse",
    action="store_true",
    help="Traverse the disjoint suffix from the final checkpoint toward the boundary.",
  )
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  args.repo = args.repo.resolve()
  args.python = args.python.resolve()
  args.protocol = args.protocol.resolve()
  args.base_checkpoint = args.base_checkpoint.resolve()
  args.training_root = args.training_root.resolve()
  args.output_root = args.output_root.resolve()
  if args.seed != EVALUATION_SEED:
    raise ValueError(f"formal evaluation seed is frozen to {EVALUATION_SEED}")
  for path in (args.python, args.protocol, args.base_checkpoint):
    if not path.is_file():
      raise FileNotFoundError(path)
  if _sha256(args.base_checkpoint) != V139_SELECTED_CHECKPOINT_SHA256:
    raise RuntimeError("prefetch base checkpoint differs from frozen v139")
  if subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=args.repo,
    check=True,
    capture_output=True,
  ).stdout:
    raise RuntimeError("prefetch requires a clean committed worktree")

  specs = _training_checkpoint_specs(args.training_root, args.base_checkpoint)
  if not 1 <= args.start_spec_index < len(specs):
    raise ValueError(
      f"start spec index must be in [1, {len(specs) - 1}]"
    )
  selected_specs = specs[args.start_spec_index :]
  if args.reverse:
    selected_specs = list(reversed(selected_specs))
  jobs = [
    (spec, condition)
    for spec in selected_specs
    for condition in FILTER_CONDITIONS
  ]
  log_root = args.output_root / "_prefetch_logs"
  log_root.mkdir(parents=True, exist_ok=True)
  progress_path = args.output_root / "prefetch_progress.json"
  durations: list[float] = []
  completed = skipped = 0
  started_unix = time.time()
  for job_index, (spec, condition) in enumerate(jobs, start=1):
    checkpoint = Path(spec["checkpoint"])
    checkpoint_sha256 = _sha256(checkpoint)
    condition_dir = _condition_dir(args.output_root, spec, condition)
    summary = _valid_existing(
      condition_dir,
      spec=spec,
      condition=condition,
      evaluation_seed=args.seed,
      checkpoint_sha256=checkpoint_sha256,
    )
    if summary is None:
      if condition_dir.exists() and any(condition_dir.iterdir()):
        raise RuntimeError(
          f"prefetch found an incomplete condition: {condition_dir}"
        )
      progress = {
        "schema_version": 1,
        "status": "running",
        "role": "disjoint_suffix_prefetch_worker",
        "start_spec_index": args.start_spec_index,
        "reverse": args.reverse,
        "selected_spec_count": len(selected_specs),
        "job_count": len(jobs),
        "job_index": job_index,
        "completed_jobs": completed,
        "skipped_jobs": skipped,
        "current_job": {
          key: (str(value) if isinstance(value, Path) else value)
          for key, value in {**spec, "filter": condition}.items()
        },
        "estimated_remaining_seconds": (
          None
          if not durations
          else (len(jobs) - completed) * sum(durations) / len(durations)
        ),
        "started_unix": started_unix,
        "updated_unix": time.time(),
      }
      _atomic_json(progress_path, progress)
      command = [
        str(args.python),
        str(args.repo / "experiments/scripts/evaluate_cbf_teacher_v31.py"),
        "--repo",
        str(args.repo),
        "--protocol",
        str(args.protocol),
        "--checkpoint",
        str(checkpoint),
        "--context",
        str(spec["context"]),
        "--runtime-filter",
        condition,
        "--num-envs",
        str(PAIRED_EVALUATION_EPISODES),
        "--num-episodes",
        str(PAIRED_EVALUATION_EPISODES),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--instrument-current-velocity-cbf",
        "--output-json",
        str(condition_dir / "summary.json"),
        "--output-csv",
        str(condition_dir / "episodes.csv"),
      ]
      environment = os.environ.copy()
      environment.update(
        {
          "PYTHONPATH": str(args.repo),
          "MUJOCO_GL": "egl",
          "PYTHONUNBUFFERED": "1",
        }
      )
      log_name = (
        f"{spec['context']}_{spec['arm']}_{spec['training_seed']}_"
        f"round_{int(spec['round']):02d}_filter_{condition}.log"
      )
      started = time.monotonic()
      with (log_root / log_name).open("w") as log_file:
        subprocess.run(
          command,
          cwd=args.repo,
          env=environment,
          check=True,
          stdout=log_file,
          stderr=subprocess.STDOUT,
        )
      durations.append(time.monotonic() - started)
      summary = _valid_existing(
        condition_dir,
        spec=spec,
        condition=condition,
        evaluation_seed=args.seed,
        checkpoint_sha256=checkpoint_sha256,
      )
      if summary is None:
        raise RuntimeError(f"prefetched evaluator output is missing: {condition_dir}")
    else:
      skipped += 1
    completed += 1

  finished = {
    "schema_version": 1,
    "status": "complete",
    "role": "disjoint_suffix_prefetch_worker",
    "start_spec_index": args.start_spec_index,
    "reverse": args.reverse,
    "selected_spec_count": len(selected_specs),
    "job_count": len(jobs),
    "completed_jobs": completed,
    "skipped_jobs": skipped,
    "mean_job_seconds": None if not durations else sum(durations) / len(durations),
    "started_unix": started_unix,
    "finished_unix": time.time(),
  }
  _atomic_json(progress_path, finished)
  print(json.dumps(finished, sort_keys=True))


if __name__ == "__main__":
  main()
