"""Independent three-training-seed audit for CBF-Guided Brief PPO v14."""

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

from online_refine_stairs import _evaluate_state


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_rows(
  root: Path,
  *,
  domain: str,
  first_seed: int,
  repeats: int,
) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  for repeat in range(repeats):
    path = root / f"{domain}-seed{first_seed + repeat}.csv"
    with path.open(newline="") as handle:
      rows.extend(csv.DictReader(handle))
  return rows


def _column(rows: list[dict[str, str]], metric: str) -> torch.Tensor:
  if metric == "success_rate":
    values = [row["success"] == "True" for row in rows]
  elif metric == "fall_rate":
    values = [row["fell"] == "True" for row in rows]
  else:
    values = [float(row[metric]) for row in rows]
  tensor = torch.tensor(values, dtype=torch.float64)
  if not bool(torch.isfinite(tensor).all()):
    raise ValueError(f"audit metric {metric} contains non-finite values")
  return tensor


def _hierarchical_interval(
  groups: list[torch.Tensor],
  *,
  bootstrap_samples: int,
  seed: int,
) -> tuple[float, float, float]:
  """Bootstrap training seeds, then episodes within sampled seeds."""
  if len(groups) < 2 or not groups:
    raise ValueError("hierarchical interval requires multiple training seeds")
  lengths = {int(group.numel()) for group in groups}
  if len(lengths) != 1 or 0 in lengths:
    raise ValueError("hierarchical groups must have one non-zero common size")
  values = torch.stack(groups)
  group_count, episode_count = values.shape
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  means: list[torch.Tensor] = []
  chunk_size = 250
  for start in range(0, bootstrap_samples, chunk_size):
    count = min(chunk_size, bootstrap_samples - start)
    sampled_groups = torch.randint(
      group_count, (count, group_count), generator=generator
    )
    sampled_episodes = torch.randint(
      episode_count,
      (count, group_count, episode_count),
      generator=generator,
    )
    selected = values[sampled_groups]
    samples = torch.gather(selected, 2, sampled_episodes)
    means.append(samples.mean(dim=(1, 2)))
  bootstrap_means = torch.cat(means)
  lower, upper = torch.quantile(
    bootstrap_means, torch.tensor([0.025, 0.975], dtype=torch.float64)
  )
  return float(values.mean()), float(lower), float(upper)


def _paired_rows(
  baseline_rows: list[dict[str, str]],
  final_rows: list[dict[str, str]],
  *,
  metric: str,
) -> torch.Tensor:
  if len(baseline_rows) != len(final_rows):
    raise ValueError("paired audit row counts differ")
  for index, (baseline, final) in enumerate(
    zip(baseline_rows, final_rows, strict=True)
  ):
    if baseline["episode"] != final["episode"]:
      raise ValueError(f"paired episode index differs at row {index}")
  return _column(final_rows, metric) - _column(baseline_rows, metric)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--baseline-checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-template", required=True)
  parser.add_argument(
    "--training-seeds", nargs="+", type=int, default=(42, 142, 242)
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--audit-seed", type=int, default=1400000)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if len(args.training_seeds) != 3 or len(set(args.training_seeds)) != 3:
    raise ValueError("formal audit requires exactly three distinct training seeds")
  if args.eval_batch_size != 128:
    raise ValueError("formal v14 audit uses 128 independent environments per batch")
  if args.bootstrap_samples < 1000:
    raise ValueError("formal audit requires at least 1000 bootstrap samples")

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  baseline_path = args.baseline_checkpoint.resolve()
  baseline_payload = torch.load(
    baseline_path, map_location="cpu", weights_only=False
  )
  baseline_actor = baseline_payload["actor_state_dict"]
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg("Unitree-G1-Stairs-Online-DQH")
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.audit_seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg("Unitree-G1-Stairs-Online-DQH")
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls("Unitree-G1-Stairs-Online-DQH")
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)

  protocol = {
    "DQH": {"episodes_per_training_seed": 512, "repeats": 4},
    "D0": {"episodes_per_training_seed": 256, "repeats": 2},
    "DQNH": {"episodes_per_training_seed": 256, "repeats": 2},
  }
  raw: dict[str, Any] = {}
  rows_by_domain: dict[str, dict[str, list[list[dict[str, str]]]]] = {
    domain: {"baseline": [], "final": []} for domain in protocol
  }
  candidate_checksums: dict[str, str] = {}
  for seed_index, training_seed in enumerate(args.training_seeds):
    candidate_path = Path(
      args.candidate_template.format(seed=training_seed)
    ).resolve()
    candidate_payload = torch.load(
      candidate_path, map_location="cpu", weights_only=False
    )
    candidate_actor = candidate_payload["actor_state_dict"]
    candidate_checksums[str(training_seed)] = _sha256(candidate_path)
    first_eval_seed = args.audit_seed + 10000 * seed_index
    seed_output = output_dir / f"train_seed{training_seed}"
    raw[str(training_seed)] = {
      "candidate_checkpoint": str(candidate_path),
      "candidate_checkpoint_sha256": candidate_checksums[str(training_seed)],
      "evaluation_seed_start": first_eval_seed,
      "domains": {},
    }
    for domain, domain_protocol in protocol.items():
      repeats = domain_protocol["repeats"]
      baseline_dir = seed_output / "baseline" / domain
      final_dir = seed_output / "final" / domain
      baseline_eval = _evaluate_state(
        runner,
        baseline_actor,
        domains=(domain,),
        num_envs=args.eval_batch_size,
        num_episodes=args.eval_batch_size,
        seed=first_eval_seed,
        device=args.device,
        repeats=repeats,
        runtime_filter=True,
        artifact_dir=baseline_dir,
        resume=True,
      )[domain]
      final_eval = _evaluate_state(
        runner,
        candidate_actor,
        domains=(domain,),
        num_envs=args.eval_batch_size,
        num_episodes=args.eval_batch_size,
        seed=first_eval_seed,
        device=args.device,
        repeats=repeats,
        runtime_filter=True,
        artifact_dir=final_dir,
        resume=True,
      )[domain]
      if (
        baseline_eval["initial_state_signatures"]
        != final_eval["initial_state_signatures"]
      ):
        raise RuntimeError(
          f"{domain} audit is not paired for training seed {training_seed}"
        )
      expected = domain_protocol["episodes_per_training_seed"]
      if baseline_eval["num_episodes"] != expected or final_eval[
        "num_episodes"
      ] != expected:
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

  intervals: dict[str, Any] = {}
  metrics = ("success_rate", "fall_rate", "intervention_per_riser")
  bootstrap_seed = args.audit_seed + 900000
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
      for seed_index, training_seed in enumerate(args.training_seeds):
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
  for domain_index, domain in enumerate(protocol):
    intervals[domain] = {}
    for metric_index, metric in enumerate(metrics):
      baseline_groups = [
        _column(rows, metric)
        for rows in rows_by_domain[domain]["baseline"]
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
          for index, seed in enumerate(args.training_seeds)
        },
      }

  target_success_interval = intervals["DQH"]["success_rate"][
    "paired_delta_mean_lcb95_ucb95"
  ]
  claim_passed = target_success_interval[1] > 0.0
  result = {
    "method": "CBF-Guided Brief PPO Refinement v14",
    "evidence_role": "independent final paper audit; never used for training gates",
    "runtime_cbf": True,
    "training_seeds": args.training_seeds,
    "protocol": protocol,
    "baseline_checkpoint": str(baseline_path),
    "baseline_checkpoint_sha256": _sha256(baseline_path),
    "candidate_checkpoint_template": args.candidate_template,
    "candidate_checkpoint_sha256": candidate_checksums,
    "bootstrap": {
      "method": "hierarchical paired bootstrap over training seeds and episodes",
      "samples": args.bootstrap_samples,
      "confidence_level": 0.95,
      "seed": bootstrap_seed,
    },
    "interval_tuple_order": ["mean", "lower_95", "upper_95"],
    "paired_episode_metrics": {
      "path": str(paired_episode_path),
      "sha256": _sha256(paired_episode_path),
      "row_count": sum(
        domain_cfg["episodes_per_training_seed"]
        for domain_cfg in protocol.values()
      )
      * len(args.training_seeds),
    },
    "confidence_intervals": intervals,
    "final_target_success_claim": {
      "criterion": "LCB95[SR_DQH(final) - SR_DQH(pi0)] > 0",
      "paired_delta_mean_lcb95_ucb95": target_success_interval,
      "passed": claim_passed,
    },
    "raw_evaluations": raw,
  }
  output_path = output_dir / "final_audit_summary.json"
  output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == "__main__":
  main()
