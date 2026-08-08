"""Freeze the first base-only candidate qualifying in one v21 context family."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from online_refine_stairs import _actor_state, _evaluate_state
from specialist_v21_protocol import (
  CALIBRATION_BATCH_SIZE,
  CALIBRATION_EPISODES,
  CONTEXT_CALIBRATION_CANDIDATE_SEEDS,
  CONTEXT_CALIBRATION_EVALUATION_SEEDS,
  CONTEXTS,
  POLICY_METHOD,
  PROTOCOL_ID,
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
  serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != serialized:
    raise RuntimeError(f"refusing to overwrite a different v21 artifact: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(serialized)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--context-id", choices=CONTEXTS, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context-output", type=Path, required=True)
  parser.add_argument("--num-episodes", type=int, default=CALIBRATION_EPISODES)
  parser.add_argument("--eval-batch-size", type=int, default=CALIBRATION_BATCH_SIZE)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--exploratory", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  context_id = args.context_id
  candidate_seeds = list(CONTEXT_CALIBRATION_CANDIDATE_SEEDS[context_id])
  evaluation_seed_base = CONTEXT_CALIBRATION_EVALUATION_SEEDS[context_id]
  if args.num_episodes % args.eval_batch_size:
    raise ValueError("v21 calibration episodes must divide into full batches")
  if not args.exploratory and (
    args.num_episodes != CALIBRATION_EPISODES
    or args.eval_batch_size != CALIBRATION_BATCH_SIZE
  ):
    raise ValueError("formal v21 calibration size differs from its freeze")

  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  tracked_clean = not _git_output(
    repo, "status", "--porcelain", "--untracked-files=no"
  )
  if current_commit != args.protocol_commit:
    raise RuntimeError("v21 calibration HEAD differs from its prospective freeze")
  try:
    protocol_relative = protocol_path.relative_to(repo)
  except ValueError as exc:
    raise RuntimeError("v21 protocol must be inside the repository") from exc
  frozen_protocol = subprocess.run(
    ["git", "show", f"{current_commit}:{protocol_relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if hashlib.sha256(frozen_protocol).hexdigest() != _sha256(protocol_path):
    raise RuntimeError("v21 calibration protocol differs from its committed blob")
  declared = protocol.get("calibration", {})
  mismatches = {
    "protocol_id": (protocol.get("protocol_id"), PROTOCOL_ID),
    "revision": (protocol.get("protocol_revision"), 0),
    "status": (
      protocol.get("status"),
      "prospectively_frozen_before_base_only_calibration_and_development",
    ),
    "randomness": (protocol.get("randomness_preflight", {}).get("passed"), True),
    "base_only": (declared.get("base_policy_only"), True),
    "adapted_policy_absent": (
      declared.get("adapted_policy_evaluations_used"),
      False,
    ),
    "first_qualifier": (
      declared.get("first_qualifying_candidate_is_frozen"),
      True,
    ),
    "candidate_seeds": (
      declared.get("candidate_seeds", {}).get(context_id),
      candidate_seeds,
    ),
    "evaluation_seed": (
      declared.get("evaluation_seed_bases", {}).get(context_id),
      evaluation_seed_base,
    ),
    "episodes": (declared.get("episodes_per_candidate"), args.num_episodes),
    "batch": (declared.get("eval_batch_size"), args.eval_batch_size),
  }
  failures = {
    key: {"declared": actual, "required": required}
    for key, (actual, required) in mismatches.items()
    if actual != required
  }
  if failures:
    raise RuntimeError(f"v21 calibration protocol mismatch: {failures}")
  if not args.exploratory and not tracked_clean:
    raise RuntimeError("formal v21 calibration requires a clean tracked worktree")

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    V19_SPECIALIST_FAILURE_TYPES,
    V21_CALIBRATION_KIND,
    V21_CONTEXT_SPECS,
    configure_v19_actor_interface,
    generate_v21_specialist_context,
    validate_calibrated_v21_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  if _sha256(checkpoint) != protocol["sealed_inputs"][
    "base_policy_checkpoint_sha256"
  ]:
    raise RuntimeError("v21 calibration base checkpoint differs from its seal")
  prototype = generate_v21_specialist_context(context_id, candidate_seeds[0])
  mode = str(V21_CONTEXT_SPECS[context_id]["mode"])
  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, prototype)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = evaluation_seed_base
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  alg_cfg = agent_cfg.algorithm
  alg_cfg.actor_learning_rate = 5.0e-6
  alg_cfg.critic_learning_rate = 1.0e-4
  alg_cfg.actor_layer_multipliers = (0.10, 0.25, 0.50, 1.0)
  alg_cfg.log_std_learning_rate = 0.0
  alg_cfg.std_scale_from_base = 0.35
  alg_cfg.pre_intervention_weight = 0.0
  alg_cfg.intervention_advantage_weight = 0.0
  alg_cfg.base_anchor_weight = 0.0
  alg_cfg.d0_retention_anchor_weight = 0.0
  alg_cfg.neighbor_retention_anchor_weight = 0.0
  alg_cfg.safe_bc_weight = 0.0
  alg_cfg.correction_distillation_weight = 0.0
  alg_cfg.task_first_constrained = False
  alg_cfg.brief_ppo_refinement = True
  alg_cfg.failure_focused_refinement = True
  alg_cfg.observable_failure_conditioned_refinement = True
  alg_cfg.actor_new_feature_count = 5
  alg_cfg.actor_new_feature_learning_rate_multiplier = 1.0
  alg_cfg.freeze_legacy_actor_input_columns = True
  alg_cfg.kl_early_stopping = True
  alg_cfg.hard_case_policy_weight = 1.0
  alg_cfg.success_counterexample_policy_weight = 1.25
  alg_cfg.matched_success_preservation_beta = 0.0
  alg_cfg.clip_param = 0.05
  alg_cfg.desired_kl = 0.003
  alg_cfg.num_learning_epochs = 1
  alg_cfg.num_mini_batches = 4
  alg_cfg.schedule = "fixed"
  alg_cfg.entropy_coef = 0.0
  alg_cfg.normalize_advantage_per_mini_batch = True
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v21 calibration task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  if (
    warm_start["zero_initialized_actor_columns"] != 5
    or warm_start["pi0_exact_preservation_proof"] is not True
  ):
    raise RuntimeError("v21 calibration did not exactly zero-expand pi0")
  base_actor = _actor_state(runner.alg.actor)

  target_failure_type = V19_SPECIALIST_FAILURE_TYPES[mode]
  minimum_purity = 0.80 if mode == "lateral" else 0.75
  maximum_second = 0.30 if mode == "lateral" else 0.20
  repeats = args.num_episodes // args.eval_batch_size
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  attempts: list[dict[str, Any]] = []
  selected_payload: dict[str, Any] | None = None
  try:
    for candidate_index, candidate_seed in enumerate(candidate_seeds):
      candidate = generate_v21_specialist_context(context_id, candidate_seed)
      candidate_dir = output_dir / f"candidate_seed{candidate_seed}"
      candidate_path = candidate_dir / "context.json"
      _write_immutable_json(candidate_path, candidate)
      evaluation_seed = evaluation_seed_base + 100 * candidate_index
      evaluation = _evaluate_state(
        runner,
        base_actor,
        domains=("DQHMED",),
        num_envs=args.eval_batch_size,
        num_episodes=args.eval_batch_size,
        seed=evaluation_seed,
        repeats=repeats,
        device=args.device,
        runtime_filter=True,
        artifact_dir=candidate_dir / "evaluation",
        resume=True,
        deployment_context=candidate_path,
        v19_context=candidate_path,
      )["DQHMED"]
      counts = {
        key: int(value)
        for key, value in evaluation["failure_type_counts"].items()
      }
      fall_count = sum(counts.values())
      fractions = {
        key: value / max(1, fall_count) for key, value in counts.items()
      }
      ordered = sorted(fractions.items(), key=lambda item: (-item[1], item[0]))
      success_rate = float(evaluation["success_rate"])
      target_fraction = fractions[target_failure_type]
      second_fraction = ordered[1][1]
      qualifies = (
        0.70 <= success_rate <= 0.85
        and fall_count >= 100
        and target_fraction >= minimum_purity
        and second_fraction <= maximum_second
      )
      attempt = {
        "context_id": context_id,
        "context_family": candidate["context_family"],
        "candidate_seed": candidate_seed,
        "candidate_severity": candidate["candidate_severity"],
        "evaluation_seed_start": evaluation_seed,
        "evaluation_seeds": evaluation["seeds"],
        "parameters_sha256": candidate["parameters_sha256"],
        "base_policy_only": True,
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "fall_rate": float(evaluation["fall_rate"]),
        "fall_count": fall_count,
        "failure_type_counts": counts,
        "failure_type_fractions": fractions,
        "target_failure_type": target_failure_type,
        "target_failure_fraction": target_fraction,
        "second_failure_fraction": second_fraction,
        "qualifies": qualifies,
      }
      attempts.append(attempt)
      progress = {
        "protocol_id": PROTOCOL_ID,
        "policy_method": POLICY_METHOD,
        "context_id": context_id,
        "mode": mode,
        "candidate_seeds": candidate_seeds,
        "attempts": attempts,
        "base_policy_only": True,
        "exploratory": args.exploratory,
      }
      (output_dir / "calibration_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n"
      )
      print(json.dumps(attempt, indent=2, sort_keys=True), flush=True)
      if qualifies:
        selected_payload = candidate
        break
  finally:
    env.close()

  if args.exploratory:
    print(json.dumps({"attempts": attempts, "context_frozen": False}, indent=2))
    return
  if selected_payload is None:
    raise RuntimeError(f"no declared v21 {context_id} candidate passed every gate")
  calibration = {
    "kind": V21_CALIBRATION_KIND,
    "v21_protocol_id": PROTOCOL_ID,
    "selection_rule": "first base-only candidate satisfying every frozen gate",
    "success_rate_bounds": [0.70, 0.85],
    "minimum_target_failure_fraction": minimum_purity,
    "maximum_second_failure_fraction": maximum_second,
    "minimum_fall_count": 100,
    "episodes_per_candidate": args.num_episodes,
    "candidate_seeds": candidate_seeds,
    "attempts": attempts,
    "selected_candidate_seed": selected_payload["calibration_candidate_seed"],
    "selected_parameters_sha256": selected_payload["parameters_sha256"],
    "base_policy_checkpoint_sha256": _sha256(checkpoint),
    "adapted_policy_evaluations_used": False,
    "prospective_protocol_file_sha256": _sha256(protocol_path),
    "prospective_protocol_git_commit": current_commit,
  }
  selected_payload["calibration"] = calibration
  selected_payload = validate_calibrated_v21_context(selected_payload)
  context_output = args.context_output.resolve()
  _write_immutable_json(context_output, selected_payload)
  summary = {
    "protocol_id": PROTOCOL_ID,
    "context_id": context_id,
    "mode": mode,
    "frozen_context": str(context_output),
    "frozen_context_file_sha256": _sha256(context_output),
    "parameters_sha256": selected_payload["parameters_sha256"],
    "selected_candidate_seed": selected_payload["calibration_candidate_seed"],
    "selected_attempt": attempts[-1],
    "adapted_policy_evaluations_used": False,
  }
  (output_dir / "calibration_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
