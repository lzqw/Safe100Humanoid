"""Run short PPO capacity points while polling RTX memory."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path


def gpu_used_mib() -> int:
  out = subprocess.check_output(
    [
      "nvidia-smi",
      "--query-gpu=memory.used",
      "--format=csv,noheader,nounits",
    ],
    text=True,
  )
  return int(out.strip().splitlines()[0])


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--python", type=Path, required=True)
  parser.add_argument("--env-counts", type=int, nargs="+", required=True)
  parser.add_argument("--iterations", type=int, default=2)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.log_dir.mkdir(parents=True, exist_ok=True)
  rows = []

  for count in args.env_counts:
    log_path = args.log_dir / f"capacity_{count}.log"
    command = [
      str(args.python.resolve()),
      "scripts/train.py",
      "Unitree-G1-Stairs-CBF",
      "--env.scene.num-envs",
      str(count),
      "--agent.max-iterations",
      str(args.iterations),
      "--agent.save-interval",
      "100",
      "--agent.seed",
      "42",
      "--agent.logger",
      "tensorboard",
      "--agent.run-name",
      f"capacity_{count}",
    ]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl", "WANDB_MODE": "disabled"})
    start = time.monotonic()
    peak = gpu_used_mib()
    with log_path.open("w") as log:
      proc = subprocess.Popen(
        command,
        cwd=args.repo,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
      )
      while proc.poll() is None:
        peak = max(peak, gpu_used_mib())
        time.sleep(0.2)
      returncode = proc.returncode
    elapsed = time.monotonic() - start
    text = log_path.read_text(errors="replace")
    fps = [int(v) for v in re.findall(r"Steps per second:\s+(\d+)", text)]
    oom = "out of memory" in text.lower() or "CUDA_ERROR_OUT_OF_MEMORY" in text
    rows.append(
      {
        "num_envs": count,
        "iterations": args.iterations,
        "returncode": returncode,
        "oom": oom,
        "elapsed_s": f"{elapsed:.3f}",
        "peak_gpu_used_mib": peak,
        "mean_steps_per_second": f"{sum(fps) / len(fps):.1f}" if fps else "",
        "last_steps_per_second": fps[-1] if fps else "",
        "log": str(log_path.resolve()),
      }
    )
    print(rows[-1], flush=True)
    if returncode != 0 or oom:
      break

  with args.output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
  main()
