"""Independent diagonal-matrix audit for the three v17 specialists."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from online_refine_stairs import _actor_state, _evaluate_state


MODES = ("lateral", "cbf", "balance")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_rows(
  root: Path, *, first_seed: int, repeats: int
) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  for repeat in range(repeats):
    path = root / f"DQHMED-seed{first_seed + repeat}.csv"
    with path.open(newline="") as handle:
      rows.extend(csv.DictReader(handle))
  return rows


def _load_d0_rows(
  root: Path, *, first_seed: int, repeats: int
) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  for repeat in range(repeats):
    path = root / f"D0-seed{first_seed + repeat}.csv"
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


def hierarchical_macro_interval(
  scene_seed_deltas: list[list[torch.Tensor]],
  *,
  bootstrap_samples: int,
  seed: int,
) -> tuple[float, float, float]:
  from src.tasks.stairs_cbf.online import hierarchical_specialist_macro_interval

  return hierarchical_specialist_macro_interval(
    scene_seed_deltas,
    bootstrap_samples=bootstrap_samples,
    bootstrap_seed=seed,
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--context-dir", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--adaptation-seeds", nargs="+", type=int, default=(42, 142, 242)
  )
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--diagonal-episodes", type=int, default=512)
  parser.add_argument("--off-diagonal-episodes", type=int, default=256)
  parser.add_argument("--d0-episodes", type=int, default=256)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--audit-seed", type=int, default=1900000)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _validate_training_artifacts(
  *, training_root: Path, contexts: dict[str, dict[str, Any]], seeds: list[int]
) -> tuple[dict[str, dict[int, Path]], dict[str, Any]]:
  checkpoints: dict[str, dict[int, Path]] = {mode: {} for mode in MODES}
  summaries: dict[str, Any] = {mode: {} for mode in MODES}
  base_file_hashes: set[str] = set()
  initial_actor_hashes: set[str] = set()
  source_manifests: set[str] = set()
  for mode in MODES:
    for seed in seeds:
      run_dir = training_root / mode / f"seed{seed}"
      summary_path = run_dir / "specialist_summary.json"
      checkpoint = run_dir / "accepted_final.pt"
      if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing v17 artifact in {run_dir}")
      summary = json.loads(summary_path.read_text())
      expected_rounds = list(range(1, 6))
      reasons: list[str] = []
      if summary.get("method") != "Failure-Mode-Conditioned Brief PPO v17":
        reasons.append("method identity differs")
      if summary.get("formal_protocol") is not True:
        reasons.append("training run is not formal")
      if summary.get("specialist_mode") != mode or summary.get("seed") != seed:
        reasons.append("mode or adaptation seed differs")
      if summary.get("runtime_cbf") is not True:
        reasons.append("runtime CBF was not enabled")
      if summary.get("raw_policy_action_for_ppo") is not True:
        reasons.append("raw policy action was not retained for PPO")
      if summary.get("independent_training_branch") is not True:
        reasons.append("training branch isolation is not attested")
      source_manifest = summary.get("source_file_sha256")
      if not isinstance(source_manifest, dict) or not source_manifest:
        reasons.append("training source-file manifest is missing")
      else:
        source_manifests.add(
          json.dumps(source_manifest, sort_keys=True, separators=(",", ":"))
        )
      if summary.get("deployment_context", {}).get("parameters_sha256") != contexts[
        mode
      ]["parameters_sha256"]:
        reasons.append("training context hash differs from frozen context")
      if [record.get("round") for record in summary.get("rounds", [])] != expected_rounds:
        reasons.append("training did not record exactly five rounds")
      if summary.get("other_specialist_training_gates") is not False:
        reasons.append("another specialist was used as a training gate")
      if summary.get("neighbor_training_gate") is not False:
        reasons.append("a neighbor domain was used as a training gate")
      if summary.get("cbf_demand_training_gate") is not False:
        reasons.append("CBF demand was used as a v17 training gate")
      mixture = summary.get("integer_start_mixture_for_64_envs")
      if mixture != {"normal": 44, "failure": 10, "success": 10}:
        reasons.append("integer 70/15/15 replay allocation differs")
      if summary.get("failure_bank", {}).get("outcome_counts") != {
        "failure": summary.get("failure_bank", {}).get("size")
      }:
        reasons.append("failure bank outcome purity failed")
      success_bank = summary.get("success_counterexample_bank", {})
      if success_bank.get("matched_entry_count") != success_bank.get("size"):
        reasons.append("success counterexample matching is incomplete")
      for round_index in expected_rounds:
        if not (run_dir / f"post_round_{round_index:03d}.pt").is_file():
          reasons.append(f"round {round_index} checkpoint is missing")
      if reasons:
        raise RuntimeError(f"v17 artifact invariant failed for {mode}/{seed}: {reasons}")
      base_file_hashes.add(summary["base_policy_checkpoint_sha256"])
      initial_actor_hashes.add(summary["initial_actor_sha256"])
      checkpoints[mode][seed] = checkpoint
      summaries[mode][seed] = {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "initial_actor_sha256": summary["initial_actor_sha256"],
        "final_actor_sha256": summary["final_actor_sha256"],
        "round_gate_acceptance": [
          bool(record["target_gate_accepted"]) for record in summary["rounds"]
        ],
      }
  if len(base_file_hashes) != 1 or len(initial_actor_hashes) != 1:
    raise RuntimeError("the nine specialists did not start from one common pi0")
  if len(source_manifests) != 1:
    raise RuntimeError("the nine specialists did not use one source manifest")
  return checkpoints, {
    "base_policy_checkpoint_sha256_values": sorted(base_file_hashes),
    "initial_actor_sha256_values": sorted(initial_actor_hashes),
    "same_base_policy_file_for_all_nine_jobs": True,
    "same_initial_actor_for_all_nine_jobs": True,
    "same_source_files_for_all_nine_jobs": True,
    "source_file_sha256": json.loads(next(iter(source_manifests))),
    "runs": summaries,
  }


def main() -> None:
  args = _parse_args()
  seeds = list(args.adaptation_seeds)
  if seeds != [42, 142, 242]:
    raise ValueError("formal v17 audit requires adaptation seeds 42, 142, 242")
  if not args.smoke:
    exact = {
      "eval_batch_size": (args.eval_batch_size, 128),
      "diagonal_episodes": (args.diagonal_episodes, 512),
      "off_diagonal_episodes": (args.off_diagonal_episodes, 256),
      "d0_episodes": (args.d0_episodes, 256),
      "bootstrap_samples": (args.bootstrap_samples, 10000),
    }
    mismatches = {
      key: {"actual": actual, "required": required}
      for key, (actual, required) in exact.items()
      if actual != required
    }
    if mismatches:
      raise ValueError(f"formal v17 audit protocol mismatch: {mismatches}")
  episode_counts = (
    args.diagonal_episodes,
    args.off_diagonal_episodes,
    args.d0_episodes,
  )
  if any(count % args.eval_batch_size for count in episode_counts):
    raise ValueError("audit episode counts must divide into full paired batches")

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import (
    load_calibrated_specialist_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  context_dir = args.context_dir.resolve()
  training_root = args.training_root.resolve()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  contexts: dict[str, dict[str, Any]] = {}
  context_paths: dict[str, Path] = {}
  for mode in MODES:
    context_path = context_dir / f"{mode}.json"
    context = load_calibrated_specialist_context(context_path)
    if context["specialist_mode"] != mode:
      raise ValueError(f"frozen {mode} context has another specialist mode")
    contexts[mode] = context
    context_paths[mode] = context_path
  checkpoints, isolation = _validate_training_artifacts(
    training_root=training_root, contexts=contexts, seeds=seeds
  )
  if isolation["base_policy_checkpoint_sha256_values"] != [_sha256(checkpoint)]:
    raise RuntimeError("audit base checkpoint differs from the common training pi0")

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

  raw: dict[str, Any] = {}
  rows: dict[str, dict[int, dict[str, dict[str, list[dict[str, str]]]]]] = {
    mode: {} for mode in MODES
  }
  for specialist_index, specialist_mode in enumerate(MODES):
    raw[specialist_mode] = {}
    for seed_index, adaptation_seed in enumerate(seeds):
      runner.load_online_checkpoint(
        str(checkpoints[specialist_mode][adaptation_seed]),
        map_location=args.device,
      )
      final_actor = _actor_state(runner.alg.actor)
      raw[specialist_mode][str(adaptation_seed)] = {}
      rows[specialist_mode][adaptation_seed] = {}
      for eval_index, eval_mode in enumerate((*MODES, "D0")):
        diagonal = eval_mode == specialist_mode
        episode_count = (
          args.d0_episodes
          if eval_mode == "D0"
          else args.diagonal_episodes
          if diagonal
          else args.off_diagonal_episodes
        )
        repeats = episode_count // args.eval_batch_size
        evaluation_seed = (
          args.audit_seed
          + 10000 * specialist_index
          + 1000 * seed_index
          + 100 * eval_index
        )
        role = (
          "diagonal_primary"
          if diagonal
          else "d0_catastrophic"
          if eval_mode == "D0"
          else "off_diagonal_report_only"
        )
        eval_root = (
          output_dir
          / "raw"
          / specialist_mode
          / f"seed{adaptation_seed}"
          / eval_mode
        )
        base_root = eval_root / "baseline"
        final_root = eval_root / "final"
        domains = ("D0",) if eval_mode == "D0" else ("DQHMED",)
        deployment_context = (
          None if eval_mode == "D0" else context_paths[eval_mode]
        )
        baseline_eval = _evaluate_state(
          runner,
          base_actor,
          domains=domains,
          num_envs=args.eval_batch_size,
          num_episodes=args.eval_batch_size,
          seed=evaluation_seed,
          repeats=repeats,
          device=args.device,
          runtime_filter=True,
          artifact_dir=base_root,
          resume=True,
          deployment_context=deployment_context,
        )[domains[0]]
        final_eval = _evaluate_state(
          runner,
          final_actor,
          domains=domains,
          num_envs=args.eval_batch_size,
          num_episodes=args.eval_batch_size,
          seed=evaluation_seed,
          repeats=repeats,
          device=args.device,
          runtime_filter=True,
          artifact_dir=final_root,
          resume=True,
          deployment_context=deployment_context,
        )[domains[0]]
        if baseline_eval["initial_state_signatures"] != final_eval[
          "initial_state_signatures"
        ]:
          raise RuntimeError("v17 audit baseline/final pairing signature differs")
        loader = _load_d0_rows if eval_mode == "D0" else _load_rows
        baseline_rows = loader(
          base_root, first_seed=evaluation_seed, repeats=repeats
        )
        final_rows = loader(
          final_root, first_seed=evaluation_seed, repeats=repeats
        )
        if len(baseline_rows) != episode_count or len(final_rows) != episode_count:
          raise RuntimeError("v17 raw paired audit row count differs")
        rows[specialist_mode][adaptation_seed][eval_mode] = {
          "baseline": baseline_rows,
          "final": final_rows,
        }
        raw[specialist_mode][str(adaptation_seed)][eval_mode] = {
          "role": role,
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
        for eval_mode in (*MODES, "D0"):
          role = (
            "diagonal_primary"
            if eval_mode == specialist_mode
            else "d0_catastrophic"
            if eval_mode == "D0"
            else "off_diagonal_report_only"
          )
          pair = rows[specialist_mode][adaptation_seed][eval_mode]
          for pair_index, (baseline, final) in enumerate(
            zip(pair["baseline"], pair["final"], strict=True)
          ):
            writer.writerow(
              {
                "specialist_mode": specialist_mode,
                "evaluation_mode": eval_mode,
                "evaluation_role": role,
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

  scene_gates: dict[str, Any] = {}
  diagonal_success_deltas: list[list[torch.Tensor]] = []
  matrix: dict[str, Any] = {}
  for specialist_mode in MODES:
    matrix[specialist_mode] = {}
    for eval_mode in (*MODES, "D0"):
      success_groups = []
      fall_groups = []
      per_seed: dict[str, Any] = {}
      for adaptation_seed in seeds:
        pair = rows[specialist_mode][adaptation_seed][eval_mode]
        success_delta = _paired_delta(
          pair["baseline"], pair["final"], "success"
        )
        fall_delta = _paired_delta(pair["baseline"], pair["final"], "fell")
        success_groups.append(success_delta)
        fall_groups.append(fall_delta)
        per_seed[str(adaptation_seed)] = {
          "baseline_success_rate": float(
            _binary_column(pair["baseline"], "success").mean()
          ),
          "final_success_rate": float(
            _binary_column(pair["final"], "success").mean()
          ),
          "paired_success_delta": float(success_delta.mean()),
          "baseline_fall_rate": float(
            _binary_column(pair["baseline"], "fell").mean()
          ),
          "final_fall_rate": float(
            _binary_column(pair["final"], "fell").mean()
          ),
          "paired_fall_delta": float(fall_delta.mean()),
        }
      matrix[specialist_mode][eval_mode] = {
        "role": (
          "diagonal_primary"
          if eval_mode == specialist_mode
          else "d0_catastrophic"
          if eval_mode == "D0"
          else "off_diagonal_report_only"
        ),
        "paired_success_delta": float(torch.cat(success_groups).mean()),
        "paired_fall_delta": float(torch.cat(fall_groups).mean()),
        "per_adaptation_seed": per_seed,
      }
    diagonal = matrix[specialist_mode][specialist_mode]
    d0 = matrix[specialist_mode]["D0"]
    seed_deltas = [
      diagonal["per_adaptation_seed"][str(seed)]["paired_success_delta"]
      for seed in seeds
    ]
    criteria = {
      "diagonal_success_gain_above_2pp": diagonal["paired_success_delta"] > 0.02,
      "at_least_two_of_three_seed_gains_positive": sum(
        delta > 0.0 for delta in seed_deltas
      )
      >= 2,
      "diagonal_fall_increase_at_most_3pp": diagonal["paired_fall_delta"] <= 0.03,
      "d0_success_drop_at_most_5pp": d0["paired_success_delta"] >= -0.05,
    }
    scene_gates[specialist_mode] = {
      "criteria": criteria,
      "passed": all(criteria.values()),
      "diagonal_paired_success_delta": diagonal["paired_success_delta"],
      "diagonal_paired_fall_delta": diagonal["paired_fall_delta"],
      "per_adaptation_seed_success_delta": seed_deltas,
      "d0_paired_success_delta": d0["paired_success_delta"],
    }
    diagonal_success_deltas.append(
      [
        _paired_delta(
          rows[specialist_mode][seed][specialist_mode]["baseline"],
          rows[specialist_mode][seed][specialist_mode]["final"],
          "success",
        )
        for seed in seeds
      ]
    )

  macro_interval = hierarchical_macro_interval(
    diagonal_success_deltas,
    bootstrap_samples=args.bootstrap_samples,
    seed=args.audit_seed + 900000,
  )
  macro_passed = macro_interval[1] > 0.0
  all_scene_gates_passed = all(gate["passed"] for gate in scene_gates.values())
  result = {
    "method": "Failure-Mode-Conditioned Brief PPO v17",
    "evidence_role": "independent final diagonal audit; never used by training gates",
    "formal_protocol": not args.smoke,
    "runtime_cbf": True,
    "adaptation_seeds": seeds,
    "audit_seed": args.audit_seed,
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
      "diagonal_paired_episodes_per_adaptation_seed": args.diagonal_episodes,
      "off_diagonal_paired_episodes_per_adaptation_seed": args.off_diagonal_episodes,
      "d0_paired_episodes_per_adaptation_seed": args.d0_episodes,
      "off_diagonal_results_are_report_only": True,
      "individual_scene_lcb_gate_used": False,
    },
    "matrix": matrix,
    "scene_gates": scene_gates,
    "all_three_scene_gates_passed": all_scene_gates_passed,
    "macro_hierarchical_bootstrap": {
      "method": "paired hierarchical bootstrap over scenes, adaptation seeds, and episodes",
      "samples": args.bootstrap_samples,
      "seed": args.audit_seed + 900000,
      "tuple_order": ["mean", "lower_95", "upper_95"],
      "paired_macro_success_delta_mean_lcb95_ucb95": macro_interval,
      "criterion": "LCB95(mean diagonal success delta across three scenes) > 0",
      "passed": macro_passed,
    },
    "final_claim_passed": all_scene_gates_passed and macro_passed,
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": sum(
        args.diagonal_episodes
        + 2 * args.off_diagonal_episodes
        + args.d0_episodes
        for _mode in MODES
        for _seed in seeds
      ),
    },
    "raw_evaluations": raw,
  }
  output_path = output_dir / "final_audit_summary.json"
  output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()
