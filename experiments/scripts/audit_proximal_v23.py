"""Fresh paired target/D0 audit for the fixed round-8 v23 actor."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from proximal_v23_io import actor_state, actor_state_sha256, file_sha256
from proximal_v23_protocol import (
  BASE_CHECKPOINT_SHA256,
  CONTEXT_FILE_SHA256,
  CONTEXT_PARAMETERS_SHA256,
  EVAL_BATCH_SIZE,
  FINAL_D0_EPISODES,
  FINAL_D0_SEED,
  FINAL_TARGET_EPISODES,
  FINAL_TARGET_SEED,
  POLICY_METHOD,
  PROTOCOL_ID,
  REPORT_BOOTSTRAP_SAMPLES,
  REPORT_BOOTSTRAP_SEEDS,
  development_gate,
  repair_regression_counts,
)
from refine_proximal_v23 import _configure_algorithm, _write_json


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--final-checkpoint", type=Path, required=True)
  parser.add_argument("--training-summary", type=Path, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--protocol", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _evaluate_proximal_state(
  runner,
  state: dict[str, torch.Tensor],
  *,
  task: str,
  num_envs: int,
  repeats: int,
  seed: int,
  device: str,
  artifact_dir: Path,
  repo: Path,
  context: Path | None,
  resume: bool,
) -> dict[str, Any]:
  artifact_dir.mkdir(parents=True, exist_ok=True)
  actor_hash = actor_state_sha256(state)
  checkpoint_payload = runner.alg.save()
  checkpoint_payload["actor_state_dict"] = {
    key: value.detach().cpu() for key, value in state.items()
  }
  checkpoint_payload.setdefault("iter", 0)
  checkpoint_payload.setdefault("infos", {})
  checkpoint = artifact_dir / "actor.pt"
  torch.save(checkpoint_payload, checkpoint)
  summaries: list[dict[str, Any]] = []
  for repeat in range(repeats):
    evaluation_seed = seed + repeat
    stem = f"{task.rsplit('-', 1)[-1]}-seed{evaluation_seed}"
    output_json = artifact_dir / f"{stem}.json"
    output_csv = artifact_dir / f"{stem}.csv"
    summary = None
    if resume and output_json.is_file() and output_csv.is_file():
      try:
        candidate = json.loads(output_json.read_text())
      except (json.JSONDecodeError, OSError):
        candidate = None
      if (
        isinstance(candidate, dict)
        and candidate.get("task") == task
        and candidate.get("seed") == evaluation_seed
        and candidate.get("num_episodes") == num_envs
        and candidate.get("actor_state_sha256") == actor_hash
      ):
        summary = candidate
    if summary is None:
      command = [
        sys.executable,
        str(repo / "experiments/scripts/evaluate_proximal_v23.py"),
        "--repo",
        str(repo),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--num-envs",
        str(num_envs),
        "--num-episodes",
        str(num_envs),
        "--seed",
        str(evaluation_seed),
        "--device",
        device,
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
      ]
      if context is not None:
        command.extend(("--deployment-context", str(context)))
      completed = subprocess.run(
        command, cwd=repo, check=False, capture_output=True, text=True
      )
      if completed.returncode != 0:
        diagnostic = "\n".join(
          (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
        )
        raise RuntimeError(
          f"isolated v23 evaluation failed for {stem}:\n{diagnostic}"
        )
      summary = json.loads(output_json.read_text())
    if summary.get("actor_state_sha256") != actor_hash:
      raise RuntimeError("isolated v23 evaluator loaded a different actor")
    summaries.append(summary)
  aggregate: dict[str, Any] = {
    "task": task,
    "num_episodes": repeats * num_envs,
    "repeats": repeats,
    "seeds": [seed + index for index in range(repeats)],
    "replicates": summaries,
    "runtime_filter": True,
    "original_observation_interface": True,
    "paired_one_initial_episode_per_env": True,
    "initial_state_signatures": [
      summary["initial_state_signature"] for summary in summaries
    ],
    "actor_state_sha256": actor_hash,
  }
  for key in (
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "mean_return",
    "intervention_per_riser",
    "correction_mean",
  ):
    values = [float(summary[key]) for summary in summaries]
    aggregate[key] = sum(values) / len(values)
    aggregate[f"{key}_std"] = (
      math.sqrt(
        sum((value - aggregate[key]) ** 2 for value in values)
        / (len(values) - 1)
      )
      if len(values) > 1
      else 0.0
    )
  return aggregate


def _read_episode_rows(root: Path, domain: str, seeds: list[int]) -> list[dict[str, Any]]:
  output: list[dict[str, Any]] = []
  for seed in seeds:
    path = root / f"{domain}-seed{seed}.csv"
    if not path.is_file():
      raise FileNotFoundError(path)
    with path.open(newline="") as handle:
      for row in csv.DictReader(handle):
        output.append(
          {
            "seed": int(row["evaluation_seed"]),
            "environment_id": int(row["environment_id"]),
            "success": row["success"].lower() == "true",
            "fell": row["fell"].lower() == "true",
            "timed_out": row["timed_out"].lower() == "true",
            "return": float(row["return"]),
            "intervention_per_riser": float(row["intervention_per_riser"]),
          }
        )
  output.sort(key=lambda row: (row["seed"], row["environment_id"]))
  return output


def _paired_rows(
  domain: str,
  baseline: list[dict[str, Any]],
  final: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  if len(baseline) != len(final):
    raise RuntimeError("paired evaluation row counts differ")
  rows = []
  for index, (old, new) in enumerate(zip(baseline, final, strict=True)):
    identity = (old["seed"], old["environment_id"])
    if identity != (new["seed"], new["environment_id"]):
      raise RuntimeError("paired evaluation identities differ")
    rows.append(
      {
        "domain": domain,
        "pair_index": index,
        "evaluation_seed": identity[0],
        "environment_id": identity[1],
        "baseline_success": old["success"],
        "final_success": new["success"],
        "baseline_fell": old["fell"],
        "final_fell": new["fell"],
        "baseline_timed_out": old["timed_out"],
        "final_timed_out": new["timed_out"],
        "baseline_return": old["return"],
        "final_return": new["return"],
        "baseline_intervention_per_riser": old["intervention_per_riser"],
        "final_intervention_per_riser": new["intervention_per_riser"],
      }
    )
  return rows


def _paired_interval(
  baseline: list[float],
  final: list[float],
  *,
  seed: int,
  samples: int,
) -> dict[str, Any]:
  old = np.asarray(baseline, dtype=np.float64)
  new = np.asarray(final, dtype=np.float64)
  if old.shape != new.shape or old.ndim != 1 or old.size < 1:
    raise ValueError("paired bootstrap vectors must be equal non-empty vectors")
  delta = new - old
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, delta.size, size=(samples, delta.size))
  means = delta[indices].mean(axis=1)
  return {
    "baseline_mean": float(old.mean()),
    "final_mean": float(new.mean()),
    "delta": float(delta.mean()),
    "paired_bootstrap_ci95": [
      float(np.quantile(means, 0.025)),
      float(np.quantile(means, 0.975)),
    ],
    "bootstrap_samples": samples,
    "bootstrap_seed": seed,
    "confidence_interval_is_gate": False,
  }


def _domain_report(
  rows: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, Any]:
  baseline_success = [bool(row["baseline_success"]) for row in rows]
  final_success = [bool(row["final_success"]) for row in rows]
  baseline_fall = [float(bool(row["baseline_fell"])) for row in rows]
  final_fall = [float(bool(row["final_fell"])) for row in rows]
  return {
    "paired_conditions": len(rows),
    "success": _paired_interval(
      [float(value) for value in baseline_success],
      [float(value) for value in final_success],
      seed=bootstrap_seed,
      samples=bootstrap_samples,
    ),
    "fall": _paired_interval(
      baseline_fall,
      final_fall,
      seed=bootstrap_seed + 1,
      samples=bootstrap_samples,
    ),
    "return": _paired_interval(
      [float(row["baseline_return"]) for row in rows],
      [float(row["final_return"]) for row in rows],
      seed=bootstrap_seed + 2,
      samples=bootstrap_samples,
    ),
    "intervention_per_riser": _paired_interval(
      [float(row["baseline_intervention_per_riser"]) for row in rows],
      [float(row["final_intervention_per_riser"]) for row in rows],
      seed=bootstrap_seed + 3,
      samples=bootstrap_samples,
    ),
    "repairs_regressions": repair_regression_counts(
      baseline_success, final_success
    ),
  }


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    raise ValueError("paired CSV cannot be empty")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  checkpoint = args.base_checkpoint.resolve()
  final_checkpoint = args.final_checkpoint.resolve()
  training_path = args.training_summary.resolve()
  context_path = args.context.resolve()
  protocol_path = None if args.protocol is None else args.protocol.resolve()
  output_dir = args.output_dir.resolve()
  for path in (
    checkpoint,
    final_checkpoint,
    training_path,
    context_path,
  ):
    if not path.is_file():
      raise FileNotFoundError(path)
  training = json.loads(training_path.read_text())
  if not args.smoke and protocol_path is None:
    raise ValueError("formal v23 audit requires the frozen protocol")
  protocol = (
    json.loads(protocol_path.read_text())
    if protocol_path is not None
    else {}
  )
  if (
    training.get("protocol_id") != PROTOCOL_ID
    or training.get("policy_method") != POLICY_METHOD
    or (not args.smoke and protocol.get("protocol_id") != PROTOCOL_ID)
    or training.get("base_checkpoint", {}).get("sha256")
    != BASE_CHECKPOINT_SHA256
    or file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256
    or training.get("context", {}).get("file_sha256")
    != CONTEXT_FILE_SHA256
    or file_sha256(context_path) != CONTEXT_FILE_SHA256
    or training.get("context", {}).get("parameters_sha256")
    != CONTEXT_PARAMETERS_SHA256
  ):
    raise RuntimeError("v23 final audit inputs differ from frozen training inputs")
  if not args.smoke and (
    len(training.get("rounds", [])) != 8
    or training.get("final_policy_rule") != "round 8 actor, never best-so-far"
    or training.get("candidate_screen_or_confirmation_count") != 0
    or training.get("performance_rollbacks") != 0
  ):
    raise RuntimeError("v23 final checkpoint is not the fixed round-8 actor")

  batch_size = 4 if args.smoke else EVAL_BATCH_SIZE
  target_episodes = 4 if args.smoke else FINAL_TARGET_EPISODES
  d0_episodes = 4 if args.smoke else FINAL_D0_EPISODES
  target_repeats = target_episodes // batch_size
  d0_repeats = d0_episodes // batch_size
  target_seed = 199_240_001 if args.smoke else FINAL_TARGET_SEED
  d0_seed = 199_240_101 if args.smoke else FINAL_D0_SEED
  bootstrap_samples = 100 if args.smoke else REPORT_BOOTSTRAP_SAMPLES

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from src.tasks.stairs_cbf.deployment_context import load_calibrated_v22_context
  from src.tasks.stairs_cbf.proximal_context import apply_cbf_proximal_context
  from src.tasks.stairs_cbf.proximal import CbfProximalRefinementRunner

  context = load_calibrated_v22_context(context_path)
  env_cfg = load_env_cfg("Unitree-G1-Stairs-Online-DQHMED")
  apply_cbf_proximal_context(env_cfg, context, role="target")
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg("Unitree-G1-Stairs-Online-DQHMED")
  agent_cfg.num_steps_per_env = 8
  _configure_algorithm(agent_cfg)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner = CbfProximalRefinementRunner(
    env, asdict(agent_cfg), log_dir=None, device=args.device
  )
  try:
    runner.load_initial_checkpoint(str(checkpoint), map_location=args.device)
    actors = {"baseline": actor_state(runner.alg.actor)}
    final_payload = torch.load(
      final_checkpoint, map_location=args.device, weights_only=False
    )
    runner.alg.actor.load_state_dict(
      final_payload["actor_state_dict"], strict=True
    )
    actors["final"] = actor_state(runner.alg.actor)
    if actor_state_sha256(actors["baseline"]) != training[
      "initial_actor_sha256"
    ]:
      raise RuntimeError("final audit baseline actor differs from training pi0")
    if actor_state_sha256(actors["final"]) != training["final_actor_sha256"]:
      raise RuntimeError("final audit actor differs from fixed training round 8")

    evaluations: dict[str, dict[str, Any]] = {}
    for role, actor in actors.items():
      evaluations[f"{role}_target"] = _evaluate_proximal_state(
        runner,
        actor,
        task="Unitree-G1-Stairs-Online-DQHMED",
        num_envs=batch_size,
        repeats=target_repeats,
        seed=target_seed,
        device=args.device,
        artifact_dir=output_dir / "raw" / f"{role}_target",
        repo=repo,
        context=context_path,
        resume=args.resume,
      )
      evaluations[f"{role}_D0"] = _evaluate_proximal_state(
        runner,
        actor,
        task="Unitree-G1-Stairs-Online-D0",
        num_envs=batch_size,
        repeats=d0_repeats,
        seed=d0_seed,
        device=args.device,
        artifact_dir=output_dir / "raw" / f"{role}_D0",
        repo=repo,
        context=None,
        resume=args.resume,
      )
  finally:
    env.close()

  for domain in ("target", "D0"):
    old = evaluations[f"baseline_{domain}"]
    new = evaluations[f"final_{domain}"]
    if old["initial_state_signatures"] != new["initial_state_signatures"]:
      raise RuntimeError(f"{domain} base/final initial conditions are not paired")
    if not (
      old["runtime_filter"]
      and new["runtime_filter"]
      and old["original_observation_interface"]
      and new["original_observation_interface"]
    ):
      raise RuntimeError(f"{domain} evaluation changed the CBF/interface")

  target_seeds = [target_seed + index for index in range(target_repeats)]
  d0_seeds = [d0_seed + index for index in range(d0_repeats)]
  paired_target = _paired_rows(
    "target",
    _read_episode_rows(output_dir / "raw/baseline_target", "DQHMED", target_seeds),
    _read_episode_rows(output_dir / "raw/final_target", "DQHMED", target_seeds),
  )
  paired_d0 = _paired_rows(
    "D0",
    _read_episode_rows(output_dir / "raw/baseline_D0", "D0", d0_seeds),
    _read_episode_rows(output_dir / "raw/final_D0", "D0", d0_seeds),
  )
  paired = paired_target + paired_d0
  _write_paired_csv(output_dir / "paired_episode_metrics.csv", paired)

  target_report = _domain_report(
    paired_target,
    bootstrap_seed=REPORT_BOOTSTRAP_SEEDS["target"],
    bootstrap_samples=bootstrap_samples,
  )
  d0_report = _domain_report(
    paired_d0,
    bootstrap_seed=REPORT_BOOTSTRAP_SEEDS["D0"],
    bootstrap_samples=bootstrap_samples,
  )
  gate = development_gate(
    target_success_delta=target_report["success"]["delta"],
    target_fall_delta=target_report["fall"]["delta"],
    d0_success_delta=d0_report["success"]["delta"],
  )
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "policy_method": POLICY_METHOD,
    "smoke": args.smoke,
    "checkpoint": {
      "path": str(final_checkpoint),
      "sha256": file_sha256(final_checkpoint),
      "actor_sha256": training["final_actor_sha256"],
      "fixed_round": 8 if not args.smoke else len(training["rounds"]),
      "performance_selected": False,
    },
    "paired_evaluation": {
      "runtime_cbf": True,
      "deterministic_policy_mean": True,
      "original_actor_observation_interface": True,
      "target_episodes": len(paired_target),
      "D0_episodes": len(paired_d0),
      "target_seed_start": target_seed,
      "D0_seed_start": d0_seed,
      "base_and_final_initial_conditions_identical": True,
      "confidence_intervals_are_report_only": True,
    },
    "target": target_report,
    "D0": d0_report,
    "development_gate": gate,
    "contact_context_run": False,
    "contact_context_rule": "eligible only if the lateral development gate passes",
    "raw_evaluations": evaluations,
  }
  _write_json(output_dir / "final_test.json", result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
