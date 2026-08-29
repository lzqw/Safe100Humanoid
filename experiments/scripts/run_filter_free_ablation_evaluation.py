"""Run paired 512-episode CBF-off/on evaluations for the formal ablation."""

from __future__ import annotations

import argparse
import hashlib
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
  CHECKPOINT_ROUNDS,
  CONTEXTS,
  EVALUATION_SEED,
  PAIRED_EVALUATION_EPISODES,
  TRAINING_SEEDS,
  V139_SELECTED_CHECKPOINT_SHA256,
)

ARM_ORDER = (
  "dual_safe_ft",
  "filter_only_ft",
  "reward_only_ft",
  "nominal_ft",
)
EVALUATION_ROUND_ORDER = (4, 1, 2)
FILTER_CONDITIONS = ("off", "on")


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


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _training_checkpoint_specs(
  training_root: Path,
  base_checkpoint: Path,
) -> list[dict[str, Any]]:
  specs: list[dict[str, Any]] = [
    {
      "context": context,
      "arm": "frozen",
      "training_seed": None,
      "round": 0,
      "checkpoint": base_checkpoint,
    }
    for context in CONTEXTS
  ]
  for round_index in EVALUATION_ROUND_ORDER:
    for arm in ARM_ORDER:
      for context in CONTEXTS:
        for training_seed in TRAINING_SEEDS:
          run_dir = training_root / context / arm / f"seed_{training_seed}"
          summary_path = run_dir / "training_summary.json"
          if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
          summary = json.loads(summary_path.read_text())
          metadata = summary.get("paper_filter_free_ablation_training") or {}
          checks = {
            "context": summary.get("context") == context,
            "seed": summary.get("seed") == training_seed,
            "arm": metadata.get("arm") == arm,
            "fixed_round_4": summary.get("primary_checkpoint_round") == 4,
            "base": (
              summary.get("base_checkpoint_sha256")
              == V139_SELECTED_CHECKPOINT_SHA256
            ),
          }
          failed = sorted(name for name, passed in checks.items() if not passed)
          if failed:
            raise RuntimeError(
              f"training summary differs for {run_dir}: failed={failed}"
            )
          checkpoint = run_dir / f"round_{round_index:02d}.pt"
          if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
          specs.append(
            {
              "context": context,
              "arm": arm,
              "training_seed": training_seed,
              "round": round_index,
              "checkpoint": checkpoint,
            }
          )
  return specs


def _condition_dir(output_root: Path, spec: dict[str, Any], condition: str) -> Path:
  context = str(spec["context"])
  arm = str(spec["arm"])
  round_name = f"round_{int(spec['round']):02d}"
  if arm == "frozen":
    return output_root / context / arm / round_name / f"filter_{condition}"
  return (
    output_root
    / context
    / arm
    / f"seed_{int(spec['training_seed'])}"
    / round_name
    / f"filter_{condition}"
  )


def _valid_existing(
  condition_dir: Path,
  *,
  spec: dict[str, Any],
  condition: str,
  evaluation_seed: int,
  checkpoint_sha256: str,
) -> dict[str, Any] | None:
  summary_path = condition_dir / "summary.json"
  episodes_path = condition_dir / "episodes.csv"
  if not summary_path.is_file() or not episodes_path.is_file():
    return None
  summary = json.loads(summary_path.read_text())
  expected = {
    "context": spec["context"],
    "seed": evaluation_seed,
    "num_envs": PAIRED_EVALUATION_EPISODES,
    "num_episodes": PAIRED_EVALUATION_EPISODES,
    "runtime_filter": condition == "on",
    "checkpoint_sha256": checkpoint_sha256,
    "deterministic_policy_mean": True,
    "one_initial_episode_per_env": True,
  }
  mismatches = {
    key: (summary.get(key), value)
    for key, value in expected.items()
    if summary.get(key) != value
  }
  if mismatches:
    raise RuntimeError(
      f"existing evaluation is not reusable: {condition_dir}, {mismatches}"
    )
  return summary


def _paired_row(
  spec: dict[str, Any],
  off: dict[str, Any],
  on: dict[str, Any],
) -> dict[str, Any]:
  if off["initial_state_signature"] != on["initial_state_signature"]:
    raise RuntimeError(f"CBF-off/on initial states differ for {spec}")
  if off["actor_deterministic_state_sha256"] != on[
    "actor_deterministic_state_sha256"
  ]:
    raise RuntimeError(f"CBF-off/on actors differ for {spec}")
  return {
    "context": spec["context"],
    "arm": spec["arm"],
    "training_seed": spec["training_seed"],
    "round": spec["round"],
    "checkpoint_sha256": off["checkpoint_sha256"],
    "actor_deterministic_state_sha256": off[
      "actor_deterministic_state_sha256"
    ],
    "initial_state_signature": off["initial_state_signature"],
    "cbf_off_success_rate": off["success_rate"],
    "cbf_on_success_rate": on["success_rate"],
    "shield_gap": on["success_rate"] - off["success_rate"],
    "cbf_off_fall_rate": off["fall_rate"],
    "cbf_on_fall_rate": on["fall_rate"],
    "cbf_off_mean_reached_riser": off["mean_reached_riser"],
    "cbf_on_mean_reached_riser": on["mean_reached_riser"],
    "cbf_off_nominal_violation_steps_per_riser": off[
      "nominal_barrier_violation_steps_per_riser"
    ],
    "cbf_on_nominal_violation_steps_per_riser": on[
      "nominal_barrier_violation_steps_per_riser"
    ],
    "cbf_off_would_intervene_fraction": off[
      "counterfactual_would_intervene_fraction"
    ],
    "cbf_on_would_intervene_fraction": on[
      "counterfactual_would_intervene_fraction"
    ],
    "cbf_off_mean_counterfactual_correction_norm": off[
      "mean_counterfactual_correction_norm"
    ],
    "cbf_on_mean_correction_norm": on["mean_correction_norm"],
    "cbf_off_toe_riser_kick_episode_rate": off["kick_episode_rate"],
    "cbf_on_toe_riser_kick_episode_rate": on["kick_episode_rate"],
  }


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
    raise RuntimeError("formal evaluation base checkpoint differs from v139")
  if _git(args.repo, "status", "--porcelain"):
    raise RuntimeError("formal paired evaluation requires a clean worktree")
  training_manifest_path = args.training_root / "training_manifest.json"
  training_progress_path = args.training_root / "training_progress.json"
  if not training_manifest_path.is_file() or not training_progress_path.is_file():
    raise FileNotFoundError("formal training manifest/progress is missing")
  training_progress = json.loads(training_progress_path.read_text())
  if training_progress.get("status") != "complete":
    raise RuntimeError("formal adaptation matrix is not complete")

  specs = _training_checkpoint_specs(args.training_root, args.base_checkpoint)
  jobs = [(spec, condition) for spec in specs for condition in FILTER_CONDITIONS]
  git_commit = _git(args.repo, "rev-parse", "HEAD")
  manifest = {
    "schema_version": 1,
    "experiment": "paper_filter_free_paired_evaluation_matrix",
    "git_commit": git_commit,
    "training_git_commit": json.loads(training_manifest_path.read_text()).get(
      "git_commit"
    ),
    "base_checkpoint_sha256": V139_SELECTED_CHECKPOINT_SHA256,
    "evaluation_seed": args.seed,
    "episodes_per_filter_condition": PAIRED_EVALUATION_EPISODES,
    "contexts": list(CONTEXTS),
    "arms": ["frozen", *ADAPTATION_ARMS],
    "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
    "shared_round_0_evaluation": True,
    "checkpoint_spec_count": len(specs),
    "evaluation_job_count": len(jobs),
    "started_unix": time.time(),
  }
  args.output_root.mkdir(parents=True, exist_ok=True)
  log_root = args.output_root / "_logs"
  log_root.mkdir(parents=True, exist_ok=True)
  _atomic_json(args.output_root / "evaluation_manifest.json", manifest)
  durations: list[float] = []
  completed_jobs = 0
  skipped_jobs = 0
  paired_rows: list[dict[str, Any]] = []
  context_signatures: dict[str, str] = {}

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
          f"incomplete evaluation directory requires review: {condition_dir}"
        )
      progress = {
        **manifest,
        "status": "running",
        "job_index": job_index,
        "completed_jobs": completed_jobs,
        "skipped_jobs": skipped_jobs,
        "current_job": {
          key: (str(value) if isinstance(value, Path) else value)
          for key, value in {**spec, "filter": condition}.items()
        },
        "estimated_remaining_seconds": (
          None
          if not durations
          else (len(jobs) - completed_jobs) * sum(durations) / len(durations)
        ),
        "updated_unix": time.time(),
      }
      _atomic_json(args.output_root / "evaluation_progress.json", progress)
      print(json.dumps(progress, sort_keys=True), flush=True)
      log_name = (
        f"{spec['context']}_{spec['arm']}_{spec['training_seed']}_"
        f"round_{int(spec['round']):02d}_filter_{condition}.log"
      )
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
      assert summary is not None
    else:
      skipped_jobs += 1
    completed_jobs += 1
    context = str(spec["context"])
    signature = str(summary["initial_state_signature"])
    if context in context_signatures and context_signatures[context] != signature:
      raise RuntimeError(f"paired initial-state signature drifted in {context}")
    context_signatures[context] = signature
    if condition == "on":
      off_dir = _condition_dir(args.output_root, spec, "off")
      off = json.loads((off_dir / "summary.json").read_text())
      paired_rows.append(_paired_row(spec, off, summary))
      _atomic_json(args.output_root / "paired_checkpoint_results.json", paired_rows)

  finished = {
    **manifest,
    "status": "complete",
    "completed_jobs": completed_jobs,
    "skipped_jobs": skipped_jobs,
    "paired_checkpoint_count": len(paired_rows),
    "context_initial_state_signatures": context_signatures,
    "mean_job_seconds": None if not durations else sum(durations) / len(durations),
    "finished_unix": time.time(),
  }
  _atomic_json(args.output_root / "evaluation_progress.json", finished)
  print(json.dumps(finished, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
