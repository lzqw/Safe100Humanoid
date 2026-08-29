"""Summarize formal filter-free deployment adaptation and paired evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARMS = (
  "frozen",
  "nominal_ft",
  "reward_only_ft",
  "filter_only_ft",
  "dual_safe_ft",
)
ADAPTATION_ARMS = ARMS[1:]
CONTEXTS = ("F1", "F2", "F3")
TRAINING_SEEDS = (201357000, 201357001, 201357002)
ROUNDS = (0, 1, 2, 4)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--evaluation-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    raise ValueError(f"cannot write an empty result table: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _mean(values: list[float]) -> float:
  if not values:
    raise ValueError("mean requires at least one value")
  return statistics.fmean(values)


def _sample_std(values: list[float]) -> float:
  return statistics.stdev(values) if len(values) > 1 else 0.0


def _training_safety(training_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  run_rows: list[dict[str, Any]] = []
  for arm in ADAPTATION_ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        path = training_root / context / arm / f"seed_{seed}" / "training_summary.json"
        if not path.is_file():
          raise FileNotFoundError(path)
        summary = json.loads(path.read_text())
        metrics = [record["metrics"] for record in summary["round_metrics"]]
        if len(metrics) != 4:
          raise RuntimeError(f"training run does not contain four rounds: {path}")
        transition_count = sum(
          int(row["training_safety_transition_count"]) for row in metrics
        )
        row = {
          "arm": arm,
          "context": context,
          "training_seed": seed,
          "training_falls": sum(int(row["rollout_fall_count"]) for row in metrics),
          "training_shield_recoveries": sum(
            int(row["training_shield_recovery_count"]) for row in metrics
          ),
          "training_safety_transition_count": transition_count,
          "training_nominal_violation_count": sum(
            int(row["training_nominal_violation_count"]) for row in metrics
          ),
          "training_executed_violation_count": sum(
            int(row["training_executed_violation_count"]) for row in metrics
          ),
          "training_minimum_nominal_barrier_margin": min(
            float(row["training_minimum_nominal_barrier_margin"])
            for row in metrics
          ),
          "training_minimum_executed_barrier_margin": min(
            float(row["training_minimum_executed_barrier_margin"])
            for row in metrics
          ),
        }
        row["training_nominal_violation_fraction"] = (
          row["training_nominal_violation_count"] / transition_count
        )
        row["training_executed_violation_fraction"] = (
          row["training_executed_violation_count"] / transition_count
        )
        run_rows.append(row)

  arm_summary: dict[str, Any] = {}
  for arm in ADAPTATION_ARMS:
    per_seed: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
      rows = [
        row
        for row in run_rows
        if row["arm"] == arm and row["training_seed"] == seed
      ]
      transitions = sum(int(row["training_safety_transition_count"]) for row in rows)
      per_seed.append(
        {
          "seed": seed,
          "falls": sum(int(row["training_falls"]) for row in rows),
          "recoveries": sum(
            int(row["training_shield_recoveries"]) for row in rows
          ),
          "nominal_violation_fraction": sum(
            int(row["training_nominal_violation_count"]) for row in rows
          )
          / transitions,
          "executed_violation_fraction": sum(
            int(row["training_executed_violation_count"]) for row in rows
          )
          / transitions,
          "minimum_nominal_barrier_margin": min(
            float(row["training_minimum_nominal_barrier_margin"])
            for row in rows
          ),
          "minimum_executed_barrier_margin": min(
            float(row["training_minimum_executed_barrier_margin"])
            for row in rows
          ),
        }
      )
    arm_summary[arm] = {
      "per_seed": per_seed,
      "training_falls_mean": _mean([float(row["falls"]) for row in per_seed]),
      "training_falls_std": _sample_std(
        [float(row["falls"]) for row in per_seed]
      ),
      "training_shield_recoveries_mean": _mean(
        [float(row["recoveries"]) for row in per_seed]
      ),
      "training_nominal_violation_fraction_mean": _mean(
        [float(row["nominal_violation_fraction"]) for row in per_seed]
      ),
      "training_executed_violation_fraction_mean": _mean(
        [float(row["executed_violation_fraction"]) for row in per_seed]
      ),
      "training_minimum_nominal_barrier_margin": min(
        float(row["minimum_nominal_barrier_margin"]) for row in per_seed
      ),
      "training_minimum_executed_barrier_margin": min(
        float(row["minimum_executed_barrier_margin"]) for row in per_seed
      ),
    }
  return run_rows, arm_summary


def main() -> None:
  args = _parse_args()
  training_root = args.training_root.resolve()
  evaluation_root = args.evaluation_root.resolve()
  output_dir = args.output_dir.resolve()
  progress_path = evaluation_root / "evaluation_progress.json"
  paired_path = evaluation_root / "paired_checkpoint_results.json"
  if not progress_path.is_file() or not paired_path.is_file():
    raise FileNotFoundError("formal paired evaluation output is incomplete")
  if json.loads(progress_path.read_text()).get("status") != "complete":
    raise RuntimeError("formal paired evaluation has not completed")
  paired = json.loads(paired_path.read_text())
  if len(paired) != 111:
    raise RuntimeError(f"expected 111 paired checkpoints, found {len(paired)}")

  frozen = {row["context"]: row for row in paired if row["arm"] == "frozen"}
  if set(frozen) != set(CONTEXTS):
    raise RuntimeError("shared frozen round-0 contexts are incomplete")
  curve_rows: list[dict[str, Any]] = []
  for arm in ADAPTATION_ARMS:
    for context in CONTEXTS:
      baseline = frozen[context]
      for seed in TRAINING_SEEDS:
        curve_rows.append(
          {
            "arm": arm,
            "context": context,
            "training_seed": seed,
            "round": 0,
            "cbf_off_success_rate": baseline["cbf_off_success_rate"],
            "cbf_on_success_rate": baseline["cbf_on_success_rate"],
            "shield_gap": baseline["shield_gap"],
            "off_improvement_from_round_0": 0.0,
            "cbf_off_fall_rate": baseline["cbf_off_fall_rate"],
            "cbf_off_mean_reached_riser": baseline[
              "cbf_off_mean_reached_riser"
            ],
            "cbf_off_nominal_violation_steps_per_riser": baseline[
              "cbf_off_nominal_violation_steps_per_riser"
            ],
            "cbf_off_would_intervene_fraction": baseline[
              "cbf_off_would_intervene_fraction"
            ],
          }
        )
        for round_index in ROUNDS[1:]:
          matches = [
            row
            for row in paired
            if row["arm"] == arm
            and row["context"] == context
            and row["training_seed"] == seed
            and row["round"] == round_index
          ]
          if len(matches) != 1:
            raise RuntimeError(
              f"paired checkpoint missing: {arm}/{context}/{seed}/{round_index}"
            )
          row = matches[0]
          curve_rows.append(
            {
              "arm": arm,
              "context": context,
              "training_seed": seed,
              "round": round_index,
              "cbf_off_success_rate": row["cbf_off_success_rate"],
              "cbf_on_success_rate": row["cbf_on_success_rate"],
              "shield_gap": row["shield_gap"],
              "off_improvement_from_round_0": (
                row["cbf_off_success_rate"] - baseline["cbf_off_success_rate"]
              ),
              "cbf_off_fall_rate": row["cbf_off_fall_rate"],
              "cbf_off_mean_reached_riser": row[
                "cbf_off_mean_reached_riser"
              ],
              "cbf_off_nominal_violation_steps_per_riser": row[
                "cbf_off_nominal_violation_steps_per_riser"
              ],
              "cbf_off_would_intervene_fraction": row[
                "cbf_off_would_intervene_fraction"
              ],
            }
          )

  training_rows, training_by_arm = _training_safety(training_root)
  final_rows = [row for row in curve_rows if row["round"] == 4]
  baseline_off_mean = _mean(
    [float(frozen[context]["cbf_off_success_rate"]) for context in CONTEXTS]
  )
  main_table: list[dict[str, Any]] = [
    {
      "arm": "frozen",
      "mean_cbf_off_success_rate": baseline_off_mean,
      "std_cbf_off_success_rate_across_training_seeds": None,
      "mean_off_improvement": 0.0,
      "std_off_improvement_across_training_seeds": None,
      "mean_shield_gap": _mean(
        [float(frozen[context]["shield_gap"]) for context in CONTEXTS]
      ),
      "mean_cbf_off_nominal_violation_steps_per_riser": _mean(
        [
          float(
            frozen[context]["cbf_off_nominal_violation_steps_per_riser"]
          )
          for context in CONTEXTS
        ]
      ),
      "training_falls_mean": None,
      "training_falls_std": None,
      "training_shield_recoveries_mean": None,
      "training_nominal_violation_fraction_mean": None,
      "training_executed_violation_fraction_mean": None,
    }
  ]
  detailed_by_arm: dict[str, Any] = {}
  for arm in ADAPTATION_ARMS:
    per_seed: list[dict[str, float | int]] = []
    for seed in TRAINING_SEEDS:
      rows = [
        row
        for row in final_rows
        if row["arm"] == arm and row["training_seed"] == seed
      ]
      if len(rows) != len(CONTEXTS):
        raise RuntimeError(f"round-4 contexts are incomplete for {arm}/{seed}")
      per_seed.append(
        {
          "seed": seed,
          "mean_cbf_off_success_rate": _mean(
            [float(row["cbf_off_success_rate"]) for row in rows]
          ),
          "mean_off_improvement": _mean(
            [float(row["off_improvement_from_round_0"]) for row in rows]
          ),
          "mean_shield_gap": _mean(
            [float(row["shield_gap"]) for row in rows]
          ),
          "mean_cbf_off_nominal_violation_steps_per_riser": _mean(
            [
              float(row["cbf_off_nominal_violation_steps_per_riser"])
              for row in rows
            ]
          ),
        }
      )
    training = training_by_arm[arm]
    table_row = {
      "arm": arm,
      "mean_cbf_off_success_rate": _mean(
        [float(row["mean_cbf_off_success_rate"]) for row in per_seed]
      ),
      "std_cbf_off_success_rate_across_training_seeds": _sample_std(
        [float(row["mean_cbf_off_success_rate"]) for row in per_seed]
      ),
      "mean_off_improvement": _mean(
        [float(row["mean_off_improvement"]) for row in per_seed]
      ),
      "std_off_improvement_across_training_seeds": _sample_std(
        [float(row["mean_off_improvement"]) for row in per_seed]
      ),
      "mean_shield_gap": _mean(
        [float(row["mean_shield_gap"]) for row in per_seed]
      ),
      "mean_cbf_off_nominal_violation_steps_per_riser": _mean(
        [
          float(row["mean_cbf_off_nominal_violation_steps_per_riser"])
          for row in per_seed
        ]
      ),
      "training_falls_mean": training["training_falls_mean"],
      "training_falls_std": training["training_falls_std"],
      "training_shield_recoveries_mean": training[
        "training_shield_recoveries_mean"
      ],
      "training_nominal_violation_fraction_mean": training[
        "training_nominal_violation_fraction_mean"
      ],
      "training_executed_violation_fraction_mean": training[
        "training_executed_violation_fraction_mean"
      ],
    }
    main_table.append(table_row)
    detailed_by_arm[arm] = {"per_seed": per_seed, "training": training}

  adaptation_table = [row for row in main_table if row["arm"] != "frozen"]
  dual = next(row for row in adaptation_table if row["arm"] == "dual_safe_ft")
  best_off = max(float(row["mean_cbf_off_success_rate"]) for row in adaptation_table)
  lowest_violation = min(
    float(row["mean_cbf_off_nominal_violation_steps_per_riser"])
    for row in adaptation_table
  )
  claim_checks = {
    "dual_safe_ft_best_cbf_off_success": math.isclose(
      float(dual["mean_cbf_off_success_rate"]), best_off, abs_tol=1.0e-12
    ),
    "dual_safe_ft_lowest_nominal_violation": math.isclose(
      float(dual["mean_cbf_off_nominal_violation_steps_per_riser"]),
      lowest_violation,
      abs_tol=1.0e-12,
    ),
  }
  result = {
    "schema_version": 1,
    "experiment": "paper_filter_free_deployment_ablation",
    "primary_metric": "round_4_cbf_off_success_rate",
    "baseline_mean_cbf_off_success_rate": baseline_off_mean,
    "main_table": main_table,
    "details_by_arm": detailed_by_arm,
    "main_claim_checks": claim_checks,
    "main_claim_supported": all(claim_checks.values()),
    "curve_row_count": len(curve_rows),
    "training_safety_run_count": len(training_rows),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_json(output_dir / "final_results.json", result)
  _write_csv(output_dir / "main_table.csv", main_table)
  _write_csv(output_dir / "learning_curves.csv", curve_rows)
  _write_csv(output_dir / "training_safety.csv", training_rows)

  lines = [
    "# Filter-free deployment ablation",
    "",
    "| Arm | Mean CBF-off | Off improvement | Shield gap | Training falls |",
    "|---|---:|---:|---:|---:|",
  ]
  for row in main_table:
    falls = (
      "—"
      if row["training_falls_mean"] is None
      else f"{float(row['training_falls_mean']):.1f}"
    )
    lines.append(
      "| {arm} | {off:.3f} | {improvement:+.3f} | {gap:+.3f} | {falls} |".format(
        arm=row["arm"],
        off=float(row["mean_cbf_off_success_rate"]),
        improvement=float(row["mean_off_improvement"]),
        gap=float(row["mean_shield_gap"]),
        falls=falls,
      )
    )
  lines.extend(
    [
      "",
      f"Main claim supported: **{result['main_claim_supported']}**.",
      "",
      "The claim is enabled only when Dual Safe-FT has both the best "
      "round-4 CBF-off success and the lowest nominal violation rate.",
    ]
  )
  (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
