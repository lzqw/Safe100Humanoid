"""Evaluate pi0...pi8 after training on the frozen unseen v21 monitor set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from audit_specialists_diagonal_v19 import _load_rows
from online_refine_stairs import _actor_state, _actor_state_sha256, _evaluate_state
from specialist_v21_protocol import (
  CONTEXT_MONITOR_SEEDS,
  CONTEXTS,
  FORMAL_EVAL_BATCH_SIZE,
  FORMAL_MONITOR_EPISODES,
  FORMAL_ROUNDS,
  PROTOCOL_ID,
  configure_v21_policy_evaluation_algorithm,
  repair_regression_rates,
)

CURVE_FIELDS = [
  "context_id",
  "method_role",
  "beta",
  "round",
  "actor_sha256",
  "success_rate",
  "fall_rate",
  "success_delta_from_pi0",
  "fall_delta_from_pi0",
  "repair_rate_from_pi0",
  "regression_rate_from_pi0",
]


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--context-id", choices=CONTEXTS, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--training-dir", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  current_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  if current_commit != args.protocol_commit:
    raise RuntimeError("v21 monitor HEAD differs from the frozen protocol")
  if not args.smoke and subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip():
    raise RuntimeError("formal v21 monitor evaluation requires clean tracked files")
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  if protocol.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("unexpected v21 monitor protocol")
  frozen = subprocess.run(
    ["git", "show", f"{current_commit}:{protocol_path.relative_to(repo)}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if hashlib.sha256(frozen).hexdigest() != _sha256(protocol_path):
    raise RuntimeError("v21 monitor protocol differs from its Git blob")

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    configure_v19_actor_interface,
    load_calibrated_v21_context,
  )

  context_id = args.context_id
  context_path = args.context.resolve()
  context = load_calibrated_v21_context(context_path)
  if context["context_id"] != context_id:
    raise RuntimeError("v21 monitor context ID differs")
  declared = protocol["sealed_inputs"]["contexts"][context_id]
  if (
    _sha256(context_path) != declared["file_sha256"]
    or context["parameters_sha256"] != declared["parameters_sha256"]
  ):
    raise RuntimeError("v21 monitor context differs from its seal")
  training_dir = args.training_dir.resolve()
  training_summary = json.loads(
    (training_dir / "specialist_summary.json").read_text()
  )
  if (
    training_summary.get("context_id") != context_id
    or training_summary.get("learning_curve_protocol", {}).get(
      "monitor_set_accessed_during_training"
    )
    is not False
    or len(training_summary.get("rounds", [])) != FORMAL_ROUNDS
  ):
    raise RuntimeError("v21 training summary cannot enter monitor evaluation")
  method_role = training_summary["method_role"]
  beta = float(training_summary["matched_success_preservation_beta"])

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, context)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = CONTEXT_MONITOR_SEEDS[context_id]
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  configure_v21_policy_evaluation_algorithm(
    agent_cfg.algorithm, matched_success_preservation_beta=beta
  )
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v21 monitor task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  episode_rows = []
  actors = []
  monitor_seed = CONTEXT_MONITOR_SEEDS[context_id]
  episodes = 8 if args.smoke else FORMAL_MONITOR_EPISODES
  batch_size = 8 if args.smoke else FORMAL_EVAL_BATCH_SIZE
  repeats = episodes // batch_size
  for round_index in range(FORMAL_ROUNDS + 1):
    checkpoint = training_dir / f"post_round_{round_index:03d}.pt"
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
    actor = _actor_state(runner.alg.actor)
    actors.append(actor)
    root = output_dir / "raw" / f"round_{round_index:03d}"
    evaluation = _evaluate_state(
      runner,
      actor,
      domains=("DQHMED",),
      num_envs=batch_size,
      num_episodes=batch_size,
      seed=monitor_seed,
      repeats=repeats,
      device=args.device,
      runtime_filter=True,
      artifact_dir=root,
      resume=True,
      deployment_context=context_path,
      v19_context=context_path,
    )["DQHMED"]
    summaries.append(evaluation)
    episode_rows.append(
      _load_rows(
        root, domain="DQHMED", first_seed=monitor_seed, repeats=repeats
      )
    )
  env.close()
  signatures = [summary["initial_state_signatures"] for summary in summaries]
  if any(value != signatures[0] for value in signatures[1:]):
    raise RuntimeError("v21 monitor checkpoints did not receive paired conditions")
  baseline_success = [row["success"] == "True" for row in episode_rows[0]]
  baseline_fall = sum(row["fell"] == "True" for row in episode_rows[0]) / episodes
  curve_rows = []
  for round_index, (actor, evaluation, rows) in enumerate(
    zip(actors, summaries, episode_rows, strict=True)
  ):
    candidate_success = [row["success"] == "True" for row in rows]
    rates = repair_regression_rates(baseline_success, candidate_success)
    curve_rows.append(
      {
        "context_id": context_id,
        "method_role": method_role,
        "beta": beta,
        "round": round_index,
        "actor_sha256": _actor_state_sha256(actor),
        "success_rate": evaluation["success_rate"],
        "fall_rate": evaluation["fall_rate"],
        "success_delta_from_pi0": evaluation["success_rate"]
        - summaries[0]["success_rate"],
        "fall_delta_from_pi0": evaluation["fall_rate"] - baseline_fall,
        "repair_rate_from_pi0": rates["repair_rate"],
        "regression_rate_from_pi0": rates["regression_rate"],
      }
    )
  curve_csv = output_dir / "unseen_monitor_curve.csv"
  with curve_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
    writer.writeheader()
    writer.writerows(curve_rows)
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "context_id": context_id,
    "method_role": method_role,
    "beta": beta,
    "monitor_seed": monitor_seed,
    "paired_conditions": episodes,
    "training_selection_diagnostics_used": False,
    "monitor_set_accessed_only_after_all_checkpoints_were_saved": True,
    "checkpoint_count": len(curve_rows),
    "curve_csv": {
      "path": str(curve_csv),
      "sha256": _sha256(curve_csv),
      "row_count": len(curve_rows),
    },
    "rows": curve_rows,
  }
  output = output_dir / "unseen_monitor_curve.json"
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
