"""Precompute a disjoint suffix of the formal hardware-proxy matrix.

The primary hardware-proxy runner remains the sole writer of the manifest,
aggregate tables, and formal progress record.  This helper only writes normal
evaluator outputs for a fixed suffix; the primary runner validates and reuses
those outputs when it reaches them.
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

from experiments.scripts.run_filter_free_hardware_proxy import (  # noqa: E402
  EPISODES,
  EVALUATION_SEED,
  _job_dir,
  _sha256,
  _specs,
  _valid_summary,
)
from src.tasks.stairs_cbf.paper_filter_free_ablation import (  # noqa: E402
  V139_SELECTED_CHECKPOINT_SHA256,
)
from src.tasks.stairs_cbf.paper_hardware_proxy import (  # noqa: E402
  ACTION_DELAY_STEPS,
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
  parser.add_argument(
    "--start-job-index",
    type=int,
    required=True,
    help="Zero-based inclusive hardware-proxy job suffix boundary.",
  )
  parser.add_argument(
    "--reverse",
    action="store_true",
    help="Traverse the suffix from the final job toward the boundary.",
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
  for path in (args.python, args.protocol, args.base_checkpoint):
    if not path.is_file():
      raise FileNotFoundError(path)
  if _sha256(args.base_checkpoint) != V139_SELECTED_CHECKPOINT_SHA256:
    raise RuntimeError("hardware-proxy prefetch requires frozen v139")
  if subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=args.repo,
    check=True,
    capture_output=True,
  ).stdout:
    raise RuntimeError("hardware-proxy prefetch requires a clean worktree")
  training_progress = args.training_root / "training_progress.json"
  if not training_progress.is_file() or json.loads(
    training_progress.read_text()
  ).get("status") != "complete":
    raise RuntimeError("formal adaptation training is incomplete")

  specs = _specs(args.training_root, args.base_checkpoint)
  jobs = [(delay, spec) for delay in ACTION_DELAY_STEPS for spec in specs]
  if not 1 <= args.start_job_index < len(jobs):
    raise ValueError(
      f"start job index must be in [1, {len(jobs) - 1}]"
    )
  selected_jobs = jobs[args.start_job_index :]
  if args.reverse:
    selected_jobs = list(reversed(selected_jobs))

  args.output_root.mkdir(parents=True, exist_ok=True)
  log_root = args.output_root / "_prefetch_logs"
  log_root.mkdir(exist_ok=True)
  progress_path = args.output_root / "proxy_prefetch_progress.json"
  durations: list[float] = []
  completed = skipped = 0
  started_unix = time.time()
  for job_index, (delay, spec) in enumerate(selected_jobs, start=1):
    checkpoint = Path(spec["checkpoint"])
    checkpoint_sha256 = _sha256(checkpoint)
    output = _job_dir(args.output_root, spec, delay)
    summary = _valid_summary(
      output / "summary.json",
      spec=spec,
      delay=delay,
      checkpoint_sha256=checkpoint_sha256,
    )
    if summary is None:
      if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"incomplete hardware-proxy result: {output}")
      progress = {
        "schema_version": 1,
        "status": "running",
        "role": "disjoint_hardware_proxy_suffix_prefetch_worker",
        "start_job_index": args.start_job_index,
        "reverse": args.reverse,
        "total_job_count": len(jobs),
        "selected_job_count": len(selected_jobs),
        "job_index": job_index,
        "completed_jobs": completed,
        "skipped_jobs": skipped,
        "current_job": {
          "delay": delay,
          "context": spec["context"],
          "arm": spec["arm"],
          "training_seed": spec["training_seed"],
        },
        "estimated_remaining_seconds": (
          None
          if not durations
          else (len(selected_jobs) - completed)
          * sum(durations)
          / len(durations)
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
        spec["context"],
        "--runtime-filter",
        "off",
        "--num-envs",
        str(EPISODES),
        "--num-episodes",
        str(EPISODES),
        "--seed",
        str(EVALUATION_SEED),
        "--device",
        args.device,
        "--hardware-proxy-action-delay-steps",
        str(delay),
        "--output-json",
        str(output / "summary.json"),
        "--output-csv",
        str(output / "episodes.csv"),
      ]
      environment = os.environ.copy()
      environment.update(
        {
          "PYTHONPATH": str(args.repo),
          "MUJOCO_GL": "egl",
          "PYTHONUNBUFFERED": "1",
        }
      )
      log = log_root / (
        f"delay_{delay}_{spec['context']}_{spec['arm']}_"
        f"{spec['training_seed']}.log"
      )
      started = time.monotonic()
      with log.open("w") as handle:
        subprocess.run(
          command,
          cwd=args.repo,
          env=environment,
          check=True,
          stdout=handle,
          stderr=subprocess.STDOUT,
        )
      durations.append(time.monotonic() - started)
      summary = _valid_summary(
        output / "summary.json",
        spec=spec,
        delay=delay,
        checkpoint_sha256=checkpoint_sha256,
      )
      if summary is None:
        raise RuntimeError(f"prefetched evaluator output is missing: {output}")
    else:
      skipped += 1
    completed += 1

  finished = {
    "schema_version": 1,
    "status": "complete",
    "role": "disjoint_hardware_proxy_suffix_prefetch_worker",
    "start_job_index": args.start_job_index,
    "reverse": args.reverse,
    "total_job_count": len(jobs),
    "selected_job_count": len(selected_jobs),
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
