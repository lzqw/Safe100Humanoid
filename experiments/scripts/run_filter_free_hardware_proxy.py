"""Evaluate final CBF-off policies under the frozen hardware-proxy bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tasks.stairs_cbf.paper_filter_free_ablation import (  # noqa: E402
  CONTEXTS,
  TRAINING_SEEDS,
  V139_SELECTED_CHECKPOINT_SHA256,
)
from src.tasks.stairs_cbf.paper_hardware_proxy import (  # noqa: E402
  ACTION_DELAY_STEPS,
  METHOD_ID,
)

ARMS = ("dual_safe_ft", "filter_only_ft", "reward_only_ft", "nominal_ft")
EVALUATION_SEED = 201357901
EPISODES = 256


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, default=REPO_ROOT)
  parser.add_argument("--python", type=Path, default=Path(sys.executable))
  parser.add_argument("--protocol", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _specs(training_root: Path, base_checkpoint: Path) -> list[dict[str, Any]]:
  specs = [
    {
      "context": context,
      "arm": "frozen",
      "training_seed": None,
      "checkpoint": base_checkpoint,
    }
    for context in CONTEXTS
  ]
  for arm in ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        run = training_root / context / arm / f"seed_{seed}"
        summary_path = run / "training_summary.json"
        if not summary_path.is_file():
          raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        metadata = summary.get("paper_filter_free_ablation_training") or {}
        if (
          summary.get("context") != context
          or summary.get("seed") != seed
          or metadata.get("arm") != arm
          or summary.get("primary_checkpoint_round") != 4
        ):
          raise RuntimeError(f"formal training summary differs: {summary_path}")
        checkpoint = run / "round_04.pt"
        if not checkpoint.is_file():
          raise FileNotFoundError(checkpoint)
        specs.append(
          {
            "context": context,
            "arm": arm,
            "training_seed": seed,
            "checkpoint": checkpoint,
          }
        )
  return specs


def _job_dir(root: Path, spec: dict[str, Any], delay: int) -> Path:
  base = root / f"delay_{delay}" / spec["context"] / spec["arm"]
  if spec["arm"] == "frozen":
    return base
  return base / f"seed_{spec['training_seed']}"


def _valid_summary(
  path: Path,
  *,
  spec: dict[str, Any],
  delay: int,
  checkpoint_sha256: str,
) -> dict[str, Any] | None:
  episodes_path = path.parent / "episodes.csv"
  if not path.is_file() or not episodes_path.is_file():
    return None
  summary = json.loads(path.read_text())
  proxy = summary.get("hardware_proxy") or {}
  expected = {
    "context": spec["context"],
    "seed": EVALUATION_SEED,
    "num_episodes": EPISODES,
    "runtime_filter": False,
    "checkpoint_sha256": checkpoint_sha256,
  }
  mismatches = {
    key: (summary.get(key), value)
    for key, value in expected.items()
    if summary.get(key) != value
  }
  if proxy.get("method_id") != METHOD_ID:
    mismatches["hardware_proxy.method_id"] = (proxy.get("method_id"), METHOD_ID)
  if proxy.get("action_delay_steps") != delay:
    mismatches["hardware_proxy.action_delay_steps"] = (
      proxy.get("action_delay_steps"),
      delay,
    )
  if mismatches:
    raise RuntimeError(f"hardware-proxy result differs: {path}, {mismatches}")
  return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  args.repo = args.repo.resolve()
  args.python = args.python.resolve()
  args.protocol = args.protocol.resolve()
  args.base_checkpoint = args.base_checkpoint.resolve()
  args.training_root = args.training_root.resolve()
  args.output_root = args.output_root.resolve()
  if _sha256(args.base_checkpoint) != V139_SELECTED_CHECKPOINT_SHA256:
    raise RuntimeError("hardware proxy requires frozen v139")
  if subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=args.repo,
    check=True,
    capture_output=True,
  ).stdout:
    raise RuntimeError("hardware proxy requires a clean committed worktree")
  progress_path = args.training_root / "training_progress.json"
  if not progress_path.is_file() or json.loads(progress_path.read_text()).get(
    "status"
  ) != "complete":
    raise RuntimeError("formal adaptation training is incomplete")
  specs = _specs(args.training_root, args.base_checkpoint)
  jobs = [(delay, spec) for delay in ACTION_DELAY_STEPS for spec in specs]
  manifest = {
    "schema_version": 1,
    "experiment": "paper_filter_free_hardware_proxy",
    "method_id": METHOD_ID,
    "evaluation_seed": EVALUATION_SEED,
    "episodes_per_policy_context_delay": EPISODES,
    "action_delay_steps": list(ACTION_DELAY_STEPS),
    "policy_spec_count": len(specs),
    "job_count": len(jobs),
    "started_unix": time.time(),
  }
  args.output_root.mkdir(parents=True, exist_ok=True)
  log_root = args.output_root / "_logs"
  log_root.mkdir(exist_ok=True)
  _atomic_json(args.output_root / "proxy_manifest.json", manifest)
  rows: list[dict[str, Any]] = []
  durations: list[float] = []
  signatures: dict[tuple[int, str], str] = {}
  completed = skipped = 0
  for index, (delay, spec) in enumerate(jobs, start=1):
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
        **manifest,
        "status": "running",
        "job_index": index,
        "completed_jobs": completed,
        "current_job": {
          "delay": delay,
          "context": spec["context"],
          "arm": spec["arm"],
          "training_seed": spec["training_seed"],
        },
        "estimated_remaining_seconds": (
          None
          if not durations
          else (len(jobs) - completed) * statistics.fmean(durations)
        ),
        "updated_unix": time.time(),
      }
      _atomic_json(args.output_root / "proxy_progress.json", progress)
      print(json.dumps(progress, sort_keys=True), flush=True)
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
      started = time.monotonic()
      log = log_root / (
        f"delay_{delay}_{spec['context']}_{spec['arm']}_"
        f"{spec['training_seed']}.log"
      )
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
      assert summary is not None
    else:
      skipped += 1
    completed += 1
    signature_key = (delay, spec["context"])
    signature = summary["initial_state_signature"]
    if signature_key in signatures and signatures[signature_key] != signature:
      raise RuntimeError(f"hardware-proxy initial states drifted: {signature_key}")
    signatures[signature_key] = signature
    rows.append(
      {
        "delay_steps": delay,
        "context": spec["context"],
        "arm": spec["arm"],
        "training_seed": spec["training_seed"],
        "success_rate": summary["success_rate"],
        "fall_rate": summary["fall_rate"],
        "mean_reached_riser": summary["mean_reached_riser"],
        "nominal_violation_steps_per_riser": summary[
          "nominal_barrier_violation_steps_per_riser"
        ],
        "would_intervene_fraction": summary[
          "counterfactual_would_intervene_fraction"
        ],
        "toe_riser_kick_episode_rate": summary["kick_episode_rate"],
      }
    )
    _write_csv(args.output_root / "proxy_policy_results.csv", rows)

  aggregate = []
  for arm in ("frozen", *ARMS):
    selected = [row for row in rows if row["arm"] == arm]
    aggregate.append(
      {
        "arm": arm,
        "mean_success_rate": statistics.fmean(
          float(row["success_rate"]) for row in selected
        ),
        "mean_fall_rate": statistics.fmean(
          float(row["fall_rate"]) for row in selected
        ),
        "mean_nominal_violation_steps_per_riser": statistics.fmean(
          float(row["nominal_violation_steps_per_riser"])
          for row in selected
        ),
        "policy_context_delay_count": len(selected),
      }
    )
  result = {
    **manifest,
    "status": "complete",
    "completed_jobs": completed,
    "skipped_jobs": skipped,
    "mean_job_seconds": None if not durations else statistics.fmean(durations),
    "aggregate": aggregate,
    "finished_unix": time.time(),
  }
  _atomic_json(args.output_root / "hardware_proxy_results.json", result)
  _atomic_json(args.output_root / "proxy_progress.json", result)
  _write_csv(args.output_root / "hardware_proxy_main_table.csv", aggregate)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
