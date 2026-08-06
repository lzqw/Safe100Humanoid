"""Prospective diagonal-only audit of the nine sealed v17 specialist actors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from audit_specialists_v17 import MODES, _validate_training_artifacts
from diagonal_audit_stats import (
  hierarchical_paired_scene_interval,
  independent_diagonal_scene_gate,
)
from online_refine_stairs import (
  _actor_state,
  _actor_state_sha256,
  _evaluate_state,
)

PROTOCOL_ID = "safe100-diagonal-specialist-v18"
FORMAL_ADAPTATION_SEEDS = [42, 142, 242]
FORMAL_AUDIT_SEED = 3_100_000
FORMAL_BOOTSTRAP_SEED = 4_000_000


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
  result = subprocess.run(
    ["git", *arguments],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _tracked_worktree_is_clean(repo: Path) -> bool:
  unstaged = subprocess.run(
    ["git", "diff", "--quiet"], cwd=repo, check=False
  )
  staged = subprocess.run(
    ["git", "diff", "--cached", "--quiet"], cwd=repo, check=False
  )
  return unstaged.returncode == 0 and staged.returncode == 0


def _load_rows(
  root: Path, *, domain: str, first_seed: int, repeats: int
) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  for repeat in range(repeats):
    path = root / f"{domain}-seed{first_seed + repeat}.csv"
    with path.open(newline="") as handle:
      rows.extend(csv.DictReader(handle))
  return rows


def _binary_column(rows: list[dict[str, str]], field: str) -> torch.Tensor:
  return torch.tensor(
    [row[field] == "True" for row in rows], dtype=torch.float64
  )


def _paired_delta(
  baseline: list[dict[str, str]], final: list[dict[str, str]], field: str
) -> torch.Tensor:
  if len(baseline) != len(final):
    raise ValueError("paired audit row counts differ")
  for index, (old, new) in enumerate(zip(baseline, final, strict=True)):
    if old["episode"] != new["episode"]:
      raise ValueError(f"paired episode index differs at row {index}")
  return _binary_column(final, field) - _binary_column(baseline, field)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--context-dir", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--training-manifest", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--adaptation-seeds", nargs="+", type=int, default=FORMAL_ADAPTATION_SEEDS
  )
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--target-episodes", type=int, default=512)
  parser.add_argument("--d0-episodes", type=int, default=256)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--audit-seed", type=int, default=FORMAL_AUDIT_SEED)
  parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _validate_protocol(
  protocol: dict[str, Any], args: argparse.Namespace
) -> None:
  expected = {
    "protocol_id": PROTOCOL_ID,
    "protocol_revision": 2,
    "status": "frozen_before_fresh_audit",
    "specialist_modes": list(MODES),
    "adaptation_seeds": FORMAL_ADAPTATION_SEEDS,
  }
  mismatches = {
    key: {"actual": protocol.get(key), "required": value}
    for key, value in expected.items()
    if protocol.get(key) != value
  }
  evaluation = protocol.get("evaluation", {})
  formal_evaluation = {
    "audit_seed": FORMAL_AUDIT_SEED,
    "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
    "eval_batch_size": 128,
    "target_paired_episodes_per_adaptation_seed": 512,
    "d0_paired_episodes_per_adaptation_seed": 256,
    "bootstrap_samples": 10000,
    "off_diagonal_evaluation": False,
    "macro_average_computed": False,
  }
  for key, value in formal_evaluation.items():
    if evaluation.get(key) != value:
      mismatches[f"evaluation.{key}"] = {
        "actual": evaluation.get(key),
        "required": value,
      }
  acceptance = protocol.get("independent_scene_acceptance", {})
  expected_acceptance = {
    "mean_diagonal_success_delta_strictly_above": 0.0,
    "minimum_positive_adaptation_seed_count": 2,
    "maximum_diagonal_fall_increase": 0.03,
    "minimum_d0_success_delta": -0.05,
    "confidence_interval_is_a_gate": False,
    "joint_three_scene_claim_defined": False,
  }
  for key, value in expected_acceptance.items():
    if acceptance.get(key) != value:
      mismatches[f"independent_scene_acceptance.{key}"] = {
        "actual": acceptance.get(key),
        "required": value,
      }
  if mismatches:
    raise ValueError(f"diagonal v18 protocol file mismatch: {mismatches}")
  if args.smoke:
    return
  runtime_values = {
    "adaptation_seeds": list(args.adaptation_seeds),
    "audit_seed": args.audit_seed,
    "bootstrap_seed": args.bootstrap_seed,
    "eval_batch_size": args.eval_batch_size,
    "target_paired_episodes_per_adaptation_seed": args.target_episodes,
    "d0_paired_episodes_per_adaptation_seed": args.d0_episodes,
    "bootstrap_samples": args.bootstrap_samples,
  }
  runtime_mismatches = {
    key: {
      "actual": value,
      "required": (
        FORMAL_ADAPTATION_SEEDS
        if key == "adaptation_seeds"
        else evaluation[key]
      ),
    }
    for key, value in runtime_values.items()
    if value
    != (
      FORMAL_ADAPTATION_SEEDS
      if key == "adaptation_seeds"
      else evaluation[key]
    )
  }
  if runtime_mismatches:
    raise ValueError(f"formal diagonal v18 runtime mismatch: {runtime_mismatches}")


def _validate_sealed_inputs(
  *,
  repo: Path,
  checkpoint: Path,
  context_paths: dict[str, Path],
  protocol: dict[str, Any],
  training_manifest_path: Path,
  isolation: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
  sealed = protocol["sealed_inputs"]
  if _sha256(checkpoint) != sealed["base_policy_checkpoint_sha256"]:
    raise RuntimeError("base checkpoint hash differs from the frozen protocol")
  if _sha256(training_manifest_path) != sealed["training_manifest_sha256"]:
    raise RuntimeError("training manifest hash differs from the frozen protocol")
  for mode, path in context_paths.items():
    if _sha256(path) != sealed["contexts"][mode]["file_sha256"]:
      raise RuntimeError(f"{mode} context file hash differs from the protocol")

  manifest = json.loads(training_manifest_path.read_text())
  if manifest.get("method") != "Failure-Mode-Conditioned Brief PPO v17":
    raise RuntimeError("sealed training manifest has another method")
  if manifest.get("adaptation_seeds") != FORMAL_ADAPTATION_SEEDS:
    raise RuntimeError("sealed training manifest has another adaptation seed set")
  if manifest.get("specialist_modes") != list(MODES):
    raise RuntimeError("sealed training manifest has another specialist mode set")
  common = manifest["common"]
  if common["base_policy_checkpoint_sha256"] != _sha256(checkpoint):
    raise RuntimeError("training manifest base hash differs from the checkpoint")
  source_mismatches: dict[str, dict[str, str]] = {}
  for relative, expected_hash in common["source_file_sha256"].items():
    source_path = repo / relative
    actual_hash = _sha256(source_path) if source_path.is_file() else "missing"
    if actual_hash != expected_hash:
      source_mismatches[relative] = {
        "actual": actual_hash,
        "expected": expected_hash,
      }
  if source_mismatches:
    raise RuntimeError(
      f"evaluation-relevant source differs from sealed v17 training: {source_mismatches}"
    )

  manifest_runs = {
    (record["specialist_mode"], int(record["adaptation_seed"])): record
    for record in manifest["runs"]
  }
  required_keys = {
    (mode, seed) for mode in MODES for seed in FORMAL_ADAPTATION_SEEDS
  }
  if set(manifest_runs) != required_keys:
    raise RuntimeError("training manifest does not contain exactly the nine runs")
  for key, expected in manifest_runs.items():
    mode, seed = key
    actual = isolation["runs"][mode][seed]
    for field in ("summary_sha256", "checkpoint_sha256", "final_actor_sha256"):
      if actual[field] != expected[field]:
        raise RuntimeError(f"sealed run mismatch for {mode}/{seed}/{field}")
    if (
      expected["deployment_context_file_sha256"]
      != sealed["contexts"][mode]["file_sha256"]
    ):
      raise RuntimeError(f"training manifest context file differs for {mode}/{seed}")
    if (
      expected["deployment_context_parameters_sha256"]
      != sealed["contexts"][mode]["parameters_sha256"]
    ):
      raise RuntimeError(f"training manifest context parameters differ for {mode}/{seed}")
  return manifest_runs


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))

  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  _validate_protocol(protocol, args)
  if any(
    count % args.eval_batch_size
    for count in (args.target_episodes, args.d0_episodes)
  ):
    raise ValueError("audit episode counts must divide into full paired batches")
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError(
      f"audit HEAD {current_commit} differs from protocol commit {args.protocol_commit}"
    )
  tracked_clean = _tracked_worktree_is_clean(repo)
  if not args.smoke and not tracked_clean:
    raise RuntimeError("formal audit requires a clean tracked worktree and index")

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.deployment_context import (
    load_calibrated_specialist_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  context_dir = args.context_dir.resolve()
  training_root = args.training_root.resolve()
  training_manifest_path = args.training_manifest.resolve()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  seeds = list(args.adaptation_seeds)

  contexts: dict[str, dict[str, Any]] = {}
  context_paths: dict[str, Path] = {}
  for mode in MODES:
    context_path = context_dir / f"{mode}.json"
    context = load_calibrated_specialist_context(context_path)
    if context["specialist_mode"] != mode:
      raise ValueError(f"frozen {mode} context has another specialist mode")
    calibration = context["calibration"]
    selected_attempts = [
      attempt
      for attempt in calibration.get("attempts", [])
      if attempt.get("candidate_seed") == calibration.get("selected_candidate_seed")
      and attempt.get("parameters_sha256") == context["parameters_sha256"]
    ]
    if len(selected_attempts) != 1:
      raise RuntimeError(f"frozen {mode} context has no unique selected attempt")
    selected_calibration = selected_attempts[0]
    calibration_checks = {
      "base_policy_only": selected_calibration.get("base_policy_only") is True,
      "success_rate_70_to_85_percent": 0.70
      <= selected_calibration["success_rate"]
      <= 0.85,
      "at_least_100_falls": selected_calibration["fall_count"] >= 100,
      "target_failure_at_least_60_percent": selected_calibration[
        "target_failure_fraction"
      ]
      >= 0.60,
      "second_failure_at_most_30_percent": selected_calibration[
        "second_failure_fraction"
      ]
      <= 0.30,
    }
    if not all(calibration_checks.values()):
      raise RuntimeError(f"frozen {mode} calibration no longer passes")
    contexts[mode] = context
    context_paths[mode] = context_path

  checkpoints, isolation = _validate_training_artifacts(
    training_root=training_root, contexts=contexts, seeds=seeds
  )
  manifest_runs = _validate_sealed_inputs(
    repo=repo,
    checkpoint=checkpoint,
    context_paths=context_paths,
    protocol=protocol,
    training_manifest_path=training_manifest_path,
    isolation=isolation,
  )

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.audit_seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("audit task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  base_actor = _actor_state(runner.alg.actor)
  base_actor_sha256 = _actor_state_sha256(base_actor)
  if base_actor_sha256 != isolation["initial_actor_sha256_values"][0]:
    raise RuntimeError("loaded base actor differs from sealed initial actor")

  rows: dict[str, dict[int, dict[str, dict[str, list[dict[str, str]]]]]] = {
    mode: {} for mode in MODES
  }
  raw: dict[str, Any] = {mode: {} for mode in MODES}
  baseline_cache: dict[
    tuple[str, int, str], tuple[dict[str, Any], list[dict[str, str]]]
  ] = {}
  for mode_index, specialist_mode in enumerate(MODES):
    for seed_index, adaptation_seed in enumerate(seeds):
      runner.load_online_checkpoint(
        str(checkpoints[specialist_mode][adaptation_seed]),
        map_location=args.device,
      )
      final_actor = _actor_state(runner.alg.actor)
      final_actor_sha256 = _actor_state_sha256(final_actor)
      if final_actor_sha256 != manifest_runs[
        (specialist_mode, adaptation_seed)
      ]["final_actor_sha256"]:
        raise RuntimeError(
          f"loaded final actor differs for {specialist_mode}/{adaptation_seed}"
        )
      rows[specialist_mode][adaptation_seed] = {}
      raw[specialist_mode][str(adaptation_seed)] = {}
      for evaluation_role in ("target_diagonal_primary", "d0_sanity"):
        is_target = evaluation_role == "target_diagonal_primary"
        domain = "DQHMED" if is_target else "D0"
        episode_count = args.target_episodes if is_target else args.d0_episodes
        repeats = episode_count // args.eval_batch_size
        evaluation_seed = (
          args.audit_seed + 10_000 * mode_index + 1_000 * seed_index
          if is_target
          else args.audit_seed + 90_000 + 1_000 * seed_index
        )
        context_path = context_paths[specialist_mode] if is_target else None
        cache_key = (
          specialist_mode if is_target else "common_d0",
          adaptation_seed,
          evaluation_role,
        )
        baseline_root = (
          output_dir
          / "raw"
          / "common_baseline"
          / (specialist_mode if is_target else "D0")
          / f"seed{adaptation_seed}"
        )
        final_root = (
          output_dir
          / "raw"
          / specialist_mode
          / f"seed{adaptation_seed}"
          / ("target" if is_target else "D0")
          / "final"
        )
        if cache_key not in baseline_cache:
          baseline_eval = _evaluate_state(
            runner,
            base_actor,
            domains=(domain,),
            num_envs=args.eval_batch_size,
            num_episodes=args.eval_batch_size,
            seed=evaluation_seed,
            repeats=repeats,
            device=args.device,
            runtime_filter=True,
            artifact_dir=baseline_root,
            resume=True,
            deployment_context=context_path,
          )[domain]
          baseline_rows = _load_rows(
            baseline_root,
            domain=domain,
            first_seed=evaluation_seed,
            repeats=repeats,
          )
          baseline_cache[cache_key] = (baseline_eval, baseline_rows)
        else:
          baseline_eval, baseline_rows = baseline_cache[cache_key]
        final_eval = _evaluate_state(
          runner,
          final_actor,
          domains=(domain,),
          num_envs=args.eval_batch_size,
          num_episodes=args.eval_batch_size,
          seed=evaluation_seed,
          repeats=repeats,
          device=args.device,
          runtime_filter=True,
          artifact_dir=final_root,
          resume=True,
          deployment_context=context_path,
        )[domain]
        if baseline_eval["initial_state_signatures"] != final_eval[
          "initial_state_signatures"
        ]:
          raise RuntimeError(
            f"paired signatures differ for {specialist_mode}/{adaptation_seed}/{domain}"
          )
        final_rows = _load_rows(
          final_root,
          domain=domain,
          first_seed=evaluation_seed,
          repeats=repeats,
        )
        if len(baseline_rows) != episode_count or len(final_rows) != episode_count:
          raise RuntimeError("fresh diagonal audit raw row count differs")
        rows[specialist_mode][adaptation_seed][evaluation_role] = {
          "baseline": baseline_rows,
          "final": final_rows,
        }
        raw[specialist_mode][str(adaptation_seed)][evaluation_role] = {
          "episode_count": episode_count,
          "evaluation_seed_start": evaluation_seed,
          "evaluation_seeds": baseline_eval["seeds"],
          "baseline": baseline_eval,
          "final": final_eval,
        }

  paired_csv = output_dir / "paired_episode_metrics.csv"
  with paired_csv.open("w", newline="") as handle:
    fieldnames = [
      "specialist_mode",
      "evaluation_mode",
      "evaluation_role",
      "adaptation_seed",
      "pair_index",
      "baseline_success",
      "final_success",
      "baseline_fell",
      "final_fell",
      "baseline_failure_type",
      "final_failure_type",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for specialist_mode in MODES:
      for adaptation_seed in seeds:
        for evaluation_role in ("target_diagonal_primary", "d0_sanity"):
          evaluation_mode = (
            specialist_mode
            if evaluation_role == "target_diagonal_primary"
            else "D0"
          )
          pair = rows[specialist_mode][adaptation_seed][evaluation_role]
          for pair_index, (baseline, final) in enumerate(
            zip(pair["baseline"], pair["final"], strict=True)
          ):
            writer.writerow(
              {
                "specialist_mode": specialist_mode,
                "evaluation_mode": evaluation_mode,
                "evaluation_role": evaluation_role,
                "adaptation_seed": adaptation_seed,
                "pair_index": pair_index,
                "baseline_success": int(baseline["success"] == "True"),
                "final_success": int(final["success"] == "True"),
                "baseline_fell": int(baseline["fell"] == "True"),
                "final_fell": int(final["fell"] == "True"),
                "baseline_failure_type": baseline["failure_type"],
                "final_failure_type": final["failure_type"],
              }
            )

  independent_claims: dict[str, Any] = {}
  for mode_index, specialist_mode in enumerate(MODES):
    per_seed: dict[str, Any] = {}
    target_success_groups: list[torch.Tensor] = []
    target_fall_groups: list[torch.Tensor] = []
    d0_success_groups: list[torch.Tensor] = []
    d0_fall_groups: list[torch.Tensor] = []
    for adaptation_seed in seeds:
      target_pair = rows[specialist_mode][adaptation_seed][
        "target_diagonal_primary"
      ]
      d0_pair = rows[specialist_mode][adaptation_seed]["d0_sanity"]
      target_success = _paired_delta(
        target_pair["baseline"], target_pair["final"], "success"
      )
      target_fall = _paired_delta(
        target_pair["baseline"], target_pair["final"], "fell"
      )
      d0_success = _paired_delta(
        d0_pair["baseline"], d0_pair["final"], "success"
      )
      d0_fall = _paired_delta(d0_pair["baseline"], d0_pair["final"], "fell")
      target_success_groups.append(target_success)
      target_fall_groups.append(target_fall)
      d0_success_groups.append(d0_success)
      d0_fall_groups.append(d0_fall)
      per_seed[str(adaptation_seed)] = {
        "target": {
          "baseline_success_rate": float(
            _binary_column(target_pair["baseline"], "success").mean()
          ),
          "final_success_rate": float(
            _binary_column(target_pair["final"], "success").mean()
          ),
          "paired_success_delta": float(target_success.mean()),
          "baseline_fall_rate": float(
            _binary_column(target_pair["baseline"], "fell").mean()
          ),
          "final_fall_rate": float(
            _binary_column(target_pair["final"], "fell").mean()
          ),
          "paired_fall_delta": float(target_fall.mean()),
        },
        "D0": {
          "baseline_success_rate": float(
            _binary_column(d0_pair["baseline"], "success").mean()
          ),
          "final_success_rate": float(
            _binary_column(d0_pair["final"], "success").mean()
          ),
          "paired_success_delta": float(d0_success.mean()),
          "baseline_fall_rate": float(
            _binary_column(d0_pair["baseline"], "fell").mean()
          ),
          "final_fall_rate": float(
            _binary_column(d0_pair["final"], "fell").mean()
          ),
          "paired_fall_delta": float(d0_fall.mean()),
        },
      }

    target_success_interval = hierarchical_paired_scene_interval(
      target_success_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + 100 * mode_index,
    )
    target_fall_interval = hierarchical_paired_scene_interval(
      target_fall_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 1,
    )
    d0_success_interval = hierarchical_paired_scene_interval(
      d0_success_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 2,
    )
    d0_fall_interval = hierarchical_paired_scene_interval(
      d0_fall_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 3,
    )
    per_seed_success_deltas = [
      per_seed[str(seed)]["target"]["paired_success_delta"] for seed in seeds
    ]
    gate = independent_diagonal_scene_gate(
      diagonal_success_delta=target_success_interval[0],
      per_seed_success_deltas=per_seed_success_deltas,
      diagonal_fall_delta=target_fall_interval[0],
      d0_success_delta=d0_success_interval[0],
    )
    independent_claims[specialist_mode] = {
      "question": (
        f"Does the sealed {specialist_mode} specialist improve success in its "
        "own frozen deployment context?"
      ),
      "claim_passed": gate["passed"],
      "gate": gate,
      "target": {
        "paired_success_delta_mean_lcb95_ucb95": target_success_interval,
        "paired_fall_delta_mean_lcb95_ucb95": target_fall_interval,
        "confidence_interval_is_report_only": True,
      },
      "D0_sanity": {
        "paired_success_delta_mean_lcb95_ucb95": d0_success_interval,
        "paired_fall_delta_mean_lcb95_ucb95": d0_fall_interval,
      },
      "per_adaptation_seed": per_seed,
    }

  passed_specialists = [
    mode for mode in MODES if independent_claims[mode]["claim_passed"]
  ]
  expected_rows = len(MODES) * len(seeds) * (
    args.target_episodes + args.d0_episodes
  )
  with paired_csv.open(newline="") as handle:
    actual_rows = sum(1 for _ in handle) - 1
  if actual_rows != expected_rows:
    raise RuntimeError(
      f"paired CSV has {actual_rows} rows; expected {expected_rows}"
    )
  result = {
    "protocol_id": PROTOCOL_ID,
    "analysis_version": "v18 diagonal-only prospective audit",
    "policy_method": "Failure-Mode-Conditioned Brief PPO v17",
    "evidence_role": (
      "fresh independent diagonal audit of sealed actors; never used by training gates"
    ),
    "formal_protocol": not args.smoke,
    "protocol_file": {
      "path": str(protocol_path),
      "sha256": _sha256(protocol_path),
      "git_commit": current_commit,
      "tracked_worktree_and_index_clean": tracked_clean,
    },
    "runtime_cbf": True,
    "adaptation_seeds": seeds,
    "audit_seed": args.audit_seed,
    "bootstrap_seed": args.bootstrap_seed,
    "contexts": {
      mode: {
        "path": str(context_paths[mode]),
        "file_sha256": _sha256(context_paths[mode]),
        "parameters_sha256": contexts[mode]["parameters_sha256"],
        "calibration": contexts[mode]["calibration"],
      }
      for mode in MODES
    },
    "training_isolation": isolation,
    "evaluation_protocol": {
      "target_paired_episodes_per_adaptation_seed": args.target_episodes,
      "d0_paired_episodes_per_adaptation_seed": args.d0_episodes,
      "off_diagonal_evaluation_performed": False,
      "macro_average_computed": False,
      "joint_three_scene_claim_defined": False,
      "individual_confidence_interval_used_as_gate": False,
      "paired_baseline_and_final_simulator_randomness": True,
      "common_D0_baseline_reused_across_specialists": True,
    },
    "independent_claims": independent_claims,
    "passed_specialists": passed_specialists,
    "failed_specialists": [mode for mode in MODES if mode not in passed_specialists],
    "joint_conclusion": {
      "defined": False,
      "all_three_required": False,
      "macro_gate_used": False,
      "note": "Each specialist claim stands or fails independently.",
    },
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": actual_rows,
    },
    "raw_evaluations": raw,
  }
  output_path = output_dir / "diagonal_audit_summary.json"
  temporary_path = output_dir / ".diagonal_audit_summary.json.tmp"
  temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary_path.replace(output_path)
  print(
    json.dumps(
      {
        "output_path": str(output_path),
        "paired_csv": result["paired_episode_metrics"],
        "independent_claims": {
          mode: {
            "claim_passed": independent_claims[mode]["claim_passed"],
            "target": independent_claims[mode]["target"],
            "gate": independent_claims[mode]["gate"],
          }
          for mode in MODES
        },
        "joint_conclusion": result["joint_conclusion"],
      },
      indent=2,
      sort_keys=True,
    )
  )
  env.close()


if __name__ == "__main__":
  main()
