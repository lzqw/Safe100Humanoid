"""Disjoint hierarchical paired audit for Failure-Focused Brief PPO v15."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch

from audit_brief_ppo_v14 import (
  _column,
  _hierarchical_interval,
  _load_rows,
  _paired_rows,
  _sha256,
)
from online_refine_stairs import _evaluate_state


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--baseline-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-template", required=True)
  parser.add_argument("--deployment-context", type=Path, required=True)
  parser.add_argument(
    "--training-seeds", nargs="+", type=int, default=(42, 142, 242)
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--audit-seed", type=int, default=1500000)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _verify_candidate_metadata(
  payload: dict[str, Any], *, training_seed: int, context_sha256: str
) -> dict[str, Any]:
  metadata = payload.get("infos", {}).get("online_refinement")
  if not isinstance(metadata, dict):
    raise ValueError("v15 checkpoint lacks online-refinement metadata")
  checks = {
    "method": metadata.get("method") == "Failure-Focused Brief PPO v15",
    "seed": metadata.get("seed") == training_seed,
    "train_domain": metadata.get("train_domain") == "DQHMED",
    "neighbor_domain": metadata.get("neighbor_domain") == "DQNHMED",
    "runtime_cbf": metadata.get("runtime_cbf") is True,
    "raw_policy_action_for_ppo": metadata.get("raw_policy_action_for_ppo") is True,
    "context_hash": metadata.get("deployment_context", {}).get(
      "parameters_sha256"
    )
    == context_sha256,
    "candidate_fractions": metadata.get("candidate_fractions") == [0.5, 1.0, 1.5],
    "candidate_paired_episodes": metadata.get("candidate_paired_episodes") == 128,
    "hard_fraction": abs(float(metadata.get("hard_case_fraction", -1.0)) - 10 / 64) < 1e-12,
    "hard_weight": abs(float(metadata.get("hard_case_policy_weight", -1.0)) - 0.75) < 1e-12,
    "five_rounds": len(metadata.get("rounds", [])) == 5,
    "dual_zero_all_rounds": all(
      float(round_record.get("dual_cbf_reward_weight", -1.0)) == 0.0
      for round_record in metadata.get("rounds", [])
    ),
  }
  if not all(checks.values()):
    raise ValueError(f"candidate is not a conforming v15 checkpoint: {checks}")
  return checks


def _decision_branch(
  *,
  target_improvement: bool,
  d0_noninferiority: bool,
  target_fall_safe: bool,
  cbf_demand_increased: bool,
) -> dict[str, str]:
  if not target_fall_safe:
    return {
      "branch": "HOLD-SAFETY",
      "interpretation": "target fall increase exceeded the predeclared 3 pp cap",
      "next_step": "do not promote; inspect paired target falls before another adaptation design",
    }
  if not target_improvement:
    return {
      "branch": "B",
      "interpretation": "target success improvement was not statistically established",
      "next_step": "classify target falls and build the bank around only the dominant failure type",
    }
  if not d0_noninferiority:
    return {
      "branch": "C",
      "interpretation": "target improved but D0 retention failed",
      "next_step": "use 80% target + 10% target-hard + 10% D0 in the same PPO objective",
    }
  if cbf_demand_increased:
    return {
      "branch": "D",
      "interpretation": "target improved and retained safety, but runtime CBF demand increased",
      "next_step": "run a separate 1--2 round CBF-internalization phase at dual weight 0.01",
    }
  return {
    "branch": "A",
    "interpretation": "the fixed-context target improvement claim passed",
    "next_step": "evaluate 3 base seeds x 5 hidden contexts before broadening the claim",
  }


def main() -> None:
  args = _parse_args()
  training_seeds = list(args.training_seeds)
  if training_seeds != [42, 142, 242]:
    raise ValueError("formal v15 audit requires adaptation seeds 42, 142, and 242")
  if args.eval_batch_size != 128:
    raise ValueError("formal v15 audit uses 128 independent environments per batch")
  if args.bootstrap_samples < 1000:
    raise ValueError("formal audit requires at least 1000 bootstrap samples")
  if args.audit_seed in training_seeds or 1000 <= args.audit_seed <= 1019:
    raise ValueError("final audit seed must be disjoint from calibration and adaptation")

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import (
    apply_frozen_deployment_context,
    load_calibrated_deployment_context,
  )

  context_path = args.deployment_context.resolve()
  context = load_calibrated_deployment_context(context_path)
  context_sha256 = context["parameters_sha256"]
  baseline_path = args.baseline_checkpoint.resolve()
  baseline_payload = torch.load(
    baseline_path, map_location="cpu", weights_only=False
  )
  baseline_actor = baseline_payload["actor_state_dict"]
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  task = "Unitree-G1-Stairs-Online-DQHMED"
  env_cfg = load_env_cfg(task)
  apply_frozen_deployment_context(env_cfg, context, role="target")
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.audit_seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)

  protocol = {
    "DQHMED": {"episodes_per_training_seed": 512, "repeats": 4, "role": "target"},
    "D0": {"episodes_per_training_seed": 256, "repeats": 2, "role": "retention"},
    "DQNHMED": {"episodes_per_training_seed": 256, "repeats": 2, "role": "neighbor_report_only"},
  }
  raw: dict[str, Any] = {}
  rows_by_domain: dict[str, dict[str, list[list[dict[str, str]]]]] = {
    domain: {"baseline": [], "final": []} for domain in protocol
  }
  candidate_checksums: dict[str, str] = {}
  candidate_protocol_checks: dict[str, dict[str, Any]] = {}
  try:
    for seed_index, training_seed in enumerate(training_seeds):
      candidate_path = Path(
        args.candidate_template.format(seed=training_seed)
      ).resolve()
      candidate_payload = torch.load(
        candidate_path, map_location="cpu", weights_only=False
      )
      candidate_actor = candidate_payload["actor_state_dict"]
      candidate_checksums[str(training_seed)] = _sha256(candidate_path)
      candidate_protocol_checks[str(training_seed)] = _verify_candidate_metadata(
        candidate_payload,
        training_seed=training_seed,
        context_sha256=context_sha256,
      )
      first_eval_seed = args.audit_seed + 10000 * seed_index
      seed_output = output_dir / f"train_seed{training_seed}"
      raw[str(training_seed)] = {
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": candidate_checksums[str(training_seed)],
        "evaluation_seed_start": first_eval_seed,
        "domains": {},
      }
      for domain, domain_protocol in protocol.items():
        repeats = int(domain_protocol["repeats"])
        baseline_dir = seed_output / "baseline" / domain
        final_dir = seed_output / "final" / domain
        evaluation_kwargs = {
          "domains": (domain,),
          "num_envs": args.eval_batch_size,
          "num_episodes": args.eval_batch_size,
          "seed": first_eval_seed,
          "device": args.device,
          "repeats": repeats,
          "runtime_filter": True,
          "resume": True,
          "deployment_context": context_path,
        }
        baseline_eval = _evaluate_state(
          runner,
          baseline_actor,
          artifact_dir=baseline_dir,
          **evaluation_kwargs,
        )[domain]
        final_eval = _evaluate_state(
          runner,
          candidate_actor,
          artifact_dir=final_dir,
          **evaluation_kwargs,
        )[domain]
        if baseline_eval["initial_state_signatures"] != final_eval[
          "initial_state_signatures"
        ]:
          raise RuntimeError(
            f"{domain} audit is not paired for training seed {training_seed}"
          )
        expected = int(domain_protocol["episodes_per_training_seed"])
        if (
          baseline_eval["num_episodes"] != expected
          or final_eval["num_episodes"] != expected
        ):
          raise RuntimeError(f"{domain} audit episode count differs from protocol")
        baseline_rows = _load_rows(
          baseline_dir,
          domain=domain,
          first_seed=first_eval_seed,
          repeats=repeats,
        )
        final_rows = _load_rows(
          final_dir,
          domain=domain,
          first_seed=first_eval_seed,
          repeats=repeats,
        )
        if len(baseline_rows) != expected or len(final_rows) != expected:
          raise RuntimeError(f"{domain} raw audit row count differs from protocol")
        rows_by_domain[domain]["baseline"].append(baseline_rows)
        rows_by_domain[domain]["final"].append(final_rows)
        raw[str(training_seed)]["domains"][domain] = {
          "baseline": baseline_eval,
          "final": final_eval,
        }
  finally:
    env.close()

  paired_episode_path = output_dir / "paired_episode_metrics.csv"
  with paired_episode_path.open("w", newline="") as handle:
    fieldnames = [
      "training_seed",
      "domain",
      "pair_index",
      "baseline_success",
      "final_success",
      "baseline_fell",
      "final_fell",
      "baseline_intervention_per_riser",
      "final_intervention_per_riser",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for domain in protocol:
      for seed_index, training_seed in enumerate(training_seeds):
        baseline_rows = rows_by_domain[domain]["baseline"][seed_index]
        final_rows = rows_by_domain[domain]["final"][seed_index]
        for pair_index, (baseline, final) in enumerate(
          zip(baseline_rows, final_rows, strict=True)
        ):
          writer.writerow(
            {
              "training_seed": training_seed,
              "domain": domain,
              "pair_index": pair_index,
              "baseline_success": int(baseline["success"] == "True"),
              "final_success": int(final["success"] == "True"),
              "baseline_fell": int(baseline["fell"] == "True"),
              "final_fell": int(final["fell"] == "True"),
              "baseline_intervention_per_riser": baseline[
                "intervention_per_riser"
              ],
              "final_intervention_per_riser": final[
                "intervention_per_riser"
              ],
            }
          )

  intervals: dict[str, Any] = {}
  metrics = ("success_rate", "fall_rate", "intervention_per_riser")
  bootstrap_seed = args.audit_seed + 900000
  for domain_index, domain in enumerate(protocol):
    intervals[domain] = {}
    for metric_index, metric in enumerate(metrics):
      baseline_groups = [
        _column(rows, metric) for rows in rows_by_domain[domain]["baseline"]
      ]
      final_groups = [
        _column(rows, metric) for rows in rows_by_domain[domain]["final"]
      ]
      delta_groups = [
        _paired_rows(baseline, final, metric=metric)
        for baseline, final in zip(
          rows_by_domain[domain]["baseline"],
          rows_by_domain[domain]["final"],
          strict=True,
        )
      ]
      local_seed = bootstrap_seed + 100 * domain_index + metric_index
      intervals[domain][metric] = {
        "baseline_mean_lcb95_ucb95": _hierarchical_interval(
          baseline_groups,
          bootstrap_samples=args.bootstrap_samples,
          seed=local_seed,
        ),
        "final_mean_lcb95_ucb95": _hierarchical_interval(
          final_groups,
          bootstrap_samples=args.bootstrap_samples,
          seed=local_seed + 10000,
        ),
        "paired_delta_mean_lcb95_ucb95": _hierarchical_interval(
          delta_groups,
          bootstrap_samples=args.bootstrap_samples,
          seed=local_seed + 20000,
        ),
        "per_training_seed": {
          str(seed): {
            "baseline": float(baseline_groups[index].mean()),
            "final": float(final_groups[index].mean()),
            "paired_delta": float(delta_groups[index].mean()),
          }
          for index, seed in enumerate(training_seeds)
        },
      }

  target_success = intervals["DQHMED"]["success_rate"][
    "paired_delta_mean_lcb95_ucb95"
  ]
  d0_success = intervals["D0"]["success_rate"][
    "paired_delta_mean_lcb95_ucb95"
  ]
  target_fall = intervals["DQHMED"]["fall_rate"][
    "paired_delta_mean_lcb95_ucb95"
  ]
  target_cbf = intervals["DQHMED"]["intervention_per_riser"][
    "paired_delta_mean_lcb95_ucb95"
  ]
  target_improvement = target_success[1] > 0.0
  d0_noninferiority = d0_success[1] + 0.05 >= 0.0
  target_fall_safe = target_fall[2] <= 0.03
  cbf_demand_increased = target_cbf[0] > 0.0
  gates = {
    "target_success_improvement": {
      "criterion": "LCB95[SR_target(final) - SR_target(pi0)] > 0",
      "paired_delta_mean_lcb95_ucb95": target_success,
      "passed": target_improvement,
    },
    "d0_success_noninferiority": {
      "criterion": "LCB95[SR_D0(final) - SR_D0(pi0) + 0.05] >= 0",
      "paired_delta_plus_margin_mean_lcb95_ucb95": [
        value + 0.05 for value in d0_success
      ],
      "passed": d0_noninferiority,
    },
    "target_fall_increase": {
      "criterion": "UCB95[Fall_target(final) - Fall_target(pi0)] <= 0.03",
      "paired_delta_mean_lcb95_ucb95": target_fall,
      "passed": target_fall_safe,
    },
  }
  result = {
    "method": "Failure-Focused Brief PPO v15",
    "claim_scope": "one fixed training-unseen and algorithm-hidden composite deployment context",
    "evidence_role": "independent final audit; never used by training or context selection",
    "runtime_cbf": True,
    "training_seeds": training_seeds,
    "protocol": protocol,
    "calibration_seed_range": [1000, 1019],
    "audit_seed": args.audit_seed,
    "deployment_context": {
      "path": str(context_path),
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context_sha256,
      "calibration": context["calibration"],
    },
    "baseline_checkpoint": str(baseline_path),
    "baseline_checkpoint_sha256": _sha256(baseline_path),
    "candidate_checkpoint_template": args.candidate_template,
    "candidate_checkpoint_sha256": candidate_checksums,
    "candidate_protocol_checks": candidate_protocol_checks,
    "bootstrap": {
      "method": "hierarchical paired bootstrap over adaptation seeds and episodes",
      "samples": args.bootstrap_samples,
      "confidence_level": 0.95,
      "seed": bootstrap_seed,
    },
    "interval_tuple_order": ["mean", "lower_95", "upper_95"],
    "paired_episode_metrics": {
      "path": str(paired_episode_path),
      "sha256": _sha256(paired_episode_path),
      "row_count": sum(
        int(domain_cfg["episodes_per_training_seed"])
        for domain_cfg in protocol.values()
      )
      * len(training_seeds),
    },
    "confidence_intervals": intervals,
    "promotion_gates": gates,
    "all_promotion_gates_passed": all(gate["passed"] for gate in gates.values()),
    "neighbor_role": "report only; DQNH-Medium is not a training or promotion gate",
    "target_cbf_demand_report": {
      "paired_delta_mean_lcb95_ucb95": target_cbf,
      "mean_increased": cbf_demand_increased,
      "promotion_gate": False,
    },
    "post_v15_decision": _decision_branch(
      target_improvement=target_improvement,
      d0_noninferiority=d0_noninferiority,
      target_fall_safe=target_fall_safe,
      cbf_demand_increased=cbf_demand_increased,
    ),
    "raw_evaluations": raw,
  }
  output_path = output_dir / "final_audit_summary.json"
  output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
