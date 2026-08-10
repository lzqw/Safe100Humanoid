"""Freeze the first base-only candidate qualifying for one pure v22 family."""

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
from specialist_v22_protocol import (
  CALIBRATION_EPISODES,
  CALIBRATION_MINIMUM_FALLS,
  CALIBRATION_MINIMUM_PURITY,
  CALIBRATION_SUCCESS_BOUNDS,
  CONTEXT_CALIBRATION_CANDIDATE_SEEDS,
  CONTEXT_CALIBRATION_EVALUATION_SEEDS,
  CONTEXTS,
  EVAL_BATCH_SIZE,
  POLICY_METHOD,
  PROTOCOL_ID,
  V22_CONTEXT_SCHEMA_VERSION,
  calibration_evaluation_seed,
  configure_v22_policy_evaluation_algorithm,
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
  rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite a different v22 artifact: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--context-id", choices=CONTEXTS, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context-output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  context_id = args.context_id
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError("v22 calibration HEAD differs from its prospective freeze")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("v22 calibration requires a clean tracked worktree")
  relative_protocol = protocol_path.relative_to(repo)
  frozen_protocol = subprocess.run(
    ["git", "show", f"{current_commit}:{relative_protocol}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if hashlib.sha256(frozen_protocol).hexdigest() != _sha256(protocol_path):
    raise RuntimeError("v22 calibration protocol differs from its committed blob")

  candidate_seeds = list(CONTEXT_CALIBRATION_CANDIDATE_SEEDS[context_id])
  evaluation_seed_base = CONTEXT_CALIBRATION_EVALUATION_SEEDS[context_id]
  calibration = protocol.get("calibration", {})
  checks = {
    "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
    "context_schema": protocol.get("context_schema_version")
    == V22_CONTEXT_SCHEMA_VERSION,
    "revision": protocol.get("protocol_revision") == 0,
    "status": protocol.get("status")
    == "prospectively_frozen_before_base_only_calibration",
    "randomness": protocol.get("randomness_preflight", {}).get("passed") is True,
    "base_only": calibration.get("base_policy_only") is True,
    "adapted_absent": calibration.get("adapted_policy_evaluations_used") is False,
    "first_qualifier": calibration.get("first_qualifying_candidate_is_frozen")
    is True,
    "candidate_seeds": calibration.get("candidate_seeds", {}).get(context_id)
    == candidate_seeds,
    "evaluation_seed": calibration.get("evaluation_seed_bases", {}).get(
      context_id
    )
    == evaluation_seed_base,
    "episodes": calibration.get("episodes_per_candidate")
    == CALIBRATION_EPISODES,
    "batch": calibration.get("eval_batch_size") == EVAL_BATCH_SIZE,
    "bounds": calibration.get("success_rate_bounds_inclusive")
    == list(CALIBRATION_SUCCESS_BOUNDS),
    "minimum_falls": calibration.get("minimum_fall_count")
    == CALIBRATION_MINIMUM_FALLS,
    "purity": calibration.get("minimum_target_failure_fraction")
    == CALIBRATION_MINIMUM_PURITY,
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"v22 calibration protocol mismatch: {failed}")

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    V19_SPECIALIST_FAILURE_TYPES,
    V22_CALIBRATION_KIND,
    V22_CONTEXT_SPECS,
    configure_v19_actor_interface,
    generate_v22_specialist_context,
    validate_calibrated_v22_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  if _sha256(checkpoint) != protocol["sealed_inputs"][
    "base_policy_checkpoint_sha256"
  ]:
    raise RuntimeError("v22 calibration base checkpoint differs from its seal")
  prototype = generate_v22_specialist_context(context_id, candidate_seeds[0])
  mode = str(V22_CONTEXT_SPECS[context_id]["mode"])
  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, prototype)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = evaluation_seed_base
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  configure_v22_policy_evaluation_algorithm(agent_cfg.algorithm)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v22 calibration task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  if (
    warm_start["zero_initialized_actor_columns"] != 5
    or warm_start["pi0_exact_preservation_proof"] is not True
  ):
    raise RuntimeError("v22 calibration did not exactly zero-expand pi0")
  base_actor = _actor_state(runner.alg.actor)

  target_failure_type = V19_SPECIALIST_FAILURE_TYPES[mode]
  repeats = CALIBRATION_EPISODES // EVAL_BATCH_SIZE
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  attempts: list[dict[str, Any]] = []
  selected_payload: dict[str, Any] | None = None
  try:
    for candidate_index, candidate_seed in enumerate(candidate_seeds):
      candidate = generate_v22_specialist_context(context_id, candidate_seed)
      candidate_dir = output_dir / f"candidate_seed{candidate_seed}"
      candidate_path = candidate_dir / "context.json"
      _write_immutable_json(candidate_path, candidate)
      evaluation_seed = calibration_evaluation_seed(context_id, candidate_index)
      evaluation = _evaluate_state(
        runner,
        base_actor,
        domains=("DQHMED",),
        num_envs=EVAL_BATCH_SIZE,
        num_episodes=EVAL_BATCH_SIZE,
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
        key: int(value) for key, value in evaluation["failure_type_counts"].items()
      }
      fall_count = sum(counts.values())
      fractions = {
        key: value / max(1, fall_count) for key, value in counts.items()
      }
      success_rate = float(evaluation["success_rate"])
      target_fraction = fractions[target_failure_type]
      qualifies = (
        CALIBRATION_SUCCESS_BOUNDS[0]
        <= success_rate
        <= CALIBRATION_SUCCESS_BOUNDS[1]
        and fall_count >= CALIBRATION_MINIMUM_FALLS
        and target_fraction >= CALIBRATION_MINIMUM_PURITY
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
        "num_episodes": CALIBRATION_EPISODES,
        "success_rate": success_rate,
        "fall_rate": float(evaluation["fall_rate"]),
        "fall_count": fall_count,
        "failure_type_counts": counts,
        "failure_type_fractions": fractions,
        "target_failure_type": target_failure_type,
        "target_failure_fraction": target_fraction,
        "qualifies": qualifies,
      }
      attempts.append(attempt)
      progress = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "policy_method": POLICY_METHOD,
        "context_id": context_id,
        "mode": mode,
        "candidate_seeds": candidate_seeds,
        "attempts": attempts,
        "base_policy_only": True,
        "prospective_protocol_file_sha256": _sha256(protocol_path),
        "prospective_protocol_git_commit": current_commit,
        "base_policy_checkpoint_sha256": _sha256(checkpoint),
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

  if selected_payload is None:
    negative = {
      "schema_version": 1,
      "protocol_id": PROTOCOL_ID,
      "policy_method": POLICY_METHOD,
      "status": "calibration_negative_no_candidate_qualified",
      "context_id": context_id,
      "mode": mode,
      "base_policy_only": True,
      "adapted_policy_evaluations_used": False,
      "candidate_seeds": candidate_seeds,
      "candidate_count_evaluated": len(attempts),
      "all_declared_candidates_evaluated": len(attempts) == len(candidate_seeds),
      "qualification": {
        "success_rate_bounds_inclusive": list(CALIBRATION_SUCCESS_BOUNDS),
        "minimum_fall_count": CALIBRATION_MINIMUM_FALLS,
        "target_failure_type": target_failure_type,
        "minimum_target_failure_fraction": CALIBRATION_MINIMUM_PURITY,
        "episodes_per_candidate": CALIBRATION_EPISODES,
      },
      "attempts": attempts,
      "prospective_protocol_file_sha256": _sha256(protocol_path),
      "prospective_protocol_git_commit": current_commit,
      "base_policy_checkpoint_sha256": _sha256(checkpoint),
      "adaptation_started": False,
      "final_test_started": False,
      "conditional_disposition": "stop_v22_before_lateral_adaptation",
    }
    _write_immutable_json(output_dir / "calibration_negative.json", negative)
    print(json.dumps(negative, indent=2, sort_keys=True), flush=True)
    raise RuntimeError(f"no declared v22 {context_id} candidate passed every gate")
  calibration_evidence = {
    "kind": V22_CALIBRATION_KIND,
    "v22_protocol_id": PROTOCOL_ID,
    "selection_rule": "first base-only candidate satisfying every frozen gate",
    "success_rate_bounds": list(CALIBRATION_SUCCESS_BOUNDS),
    "minimum_target_failure_fraction": CALIBRATION_MINIMUM_PURITY,
    "minimum_fall_count": CALIBRATION_MINIMUM_FALLS,
    "episodes_per_candidate": CALIBRATION_EPISODES,
    "candidate_seeds": candidate_seeds,
    "attempts": attempts,
    "selected_candidate_seed": selected_payload["calibration_candidate_seed"],
    "selected_parameters_sha256": selected_payload["parameters_sha256"],
    "base_policy_checkpoint_sha256": _sha256(checkpoint),
    "adapted_policy_evaluations_used": False,
    "prospective_protocol_file_sha256": _sha256(protocol_path),
    "prospective_protocol_git_commit": current_commit,
  }
  selected_payload["calibration"] = calibration_evidence
  selected_payload = validate_calibrated_v22_context(selected_payload)
  context_output = args.context_output.resolve()
  _write_immutable_json(context_output, selected_payload)
  summary = {
    "schema_version": 1,
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
