"""Summarize formal filter-free deployment adaptation and paired evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

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
TRAINING_ROUNDS = (1, 2, 3, 4)
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
TRANSITIONS_PER_ROUND = NUM_ENVS * ROLLOUT_STEPS
PAIRED_EPISODES = 512
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 201_357_950
ARM_FACTORS = {
  "frozen": ("—", "—"),
  "nominal_ft": ("Off", "No"),
  "reward_only_ft": ("Off", "Yes"),
  "filter_only_ft": ("On", "No"),
  "dual_safe_ft": ("On", "Yes"),
}
ARM_LABELS = {
  "frozen": "Frozen",
  "nominal_ft": "Nominal FT",
  "reward_only_ft": "Reward-only FT",
  "filter_only_ft": "Filter-only FT",
  "dual_safe_ft": "Dual Safe-FT",
}


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


def _parse_bool(value: str) -> bool:
  normalized = value.strip().lower()
  if normalized in {"true", "1"}:
    return True
  if normalized in {"false", "0"}:
    return False
  raise ValueError(f"invalid CSV boolean: {value!r}")


def _episode_success(path: Path) -> tuple[list[tuple[int, int]], np.ndarray]:
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
  keys = [
    (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
  ]
  if len(rows) != PAIRED_EPISODES or len(keys) != len(set(keys)):
    raise RuntimeError(f"paired episode identity differs: {path}")
  order = np.argsort(np.asarray(keys, dtype=np.int64)[:, 1], kind="stable")
  ordered_keys = [keys[int(index)] for index in order]
  values = np.asarray(
    [float(_parse_bool(rows[int(index)]["success"])) for index in order],
    dtype=np.float64,
  )
  return ordered_keys, values


def _episode_path(
  evaluation_root: Path,
  *,
  context: str,
  arm: str,
  training_seed: int | None,
  round_index: int,
  condition: str,
) -> Path:
  base = evaluation_root / context / arm
  if arm != "frozen":
    if training_seed is None:
      raise ValueError("adapted episode path requires a training seed")
    base = base / f"seed_{training_seed}"
  return base / f"round_{round_index:02d}" / f"filter_{condition}" / "episodes.csv"


def _hierarchical_paired_interval(
  deltas: np.ndarray,
  *,
  bootstrap_seed: int,
) -> dict[str, Any]:
  values = np.asarray(deltas, dtype=np.float64)
  expected_shape = (len(TRAINING_SEEDS), len(CONTEXTS), PAIRED_EPISODES)
  if values.shape != expected_shape or not bool(np.isfinite(values).all()):
    raise ValueError(
      f"hierarchical paired deltas differ: {values.shape} != {expected_shape}"
    )
  rng = np.random.default_rng(bootstrap_seed)
  means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
  chunk_size = 100
  seed_count, context_count, episode_count = values.shape
  for start in range(0, BOOTSTRAP_SAMPLES, chunk_size):
    stop = min(start + chunk_size, BOOTSTRAP_SAMPLES)
    count = stop - start
    sampled_seeds = rng.integers(0, seed_count, size=(count, seed_count))
    chunk_means = np.zeros(count, dtype=np.float64)
    for seed_slot in range(seed_count):
      for context_index in range(context_count):
        source = values[sampled_seeds[:, seed_slot], context_index, :]
        episode_ids = rng.integers(
          0, episode_count, size=(count, episode_count)
        )
        sampled = np.take_along_axis(source, episode_ids, axis=1)
        chunk_means += sampled.mean(axis=1) / (seed_count * context_count)
    means[start:stop] = chunk_means
  lower, upper = np.quantile(means, [0.025, 0.975])
  return {
    "mean": float(values.mean()),
    "paired_bootstrap_ci95": [float(lower), float(upper)],
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "bootstrap_seed": bootstrap_seed,
    "hierarchy": "adaptation seeds, then paired episodes; contexts fixed/equal",
    "confidence_interval_is_gate": False,
    "paired_episode_comparisons": int(values.size),
    "repairs": int(np.count_nonzero(values > 0.0)),
    "regressions": int(np.count_nonzero(values < 0.0)),
    "ties": int(np.count_nonzero(values == 0.0)),
  }


def _paired_statistics(
  evaluation_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  results: dict[str, Any] = {}
  csv_rows: list[dict[str, Any]] = []
  for arm_index, arm in enumerate(ADAPTATION_ARMS):
    improvement = np.empty(
      (len(TRAINING_SEEDS), len(CONTEXTS), PAIRED_EPISODES),
      dtype=np.float64,
    )
    shield_gap = np.empty_like(improvement)
    for seed_index, training_seed in enumerate(TRAINING_SEEDS):
      for context_index, context in enumerate(CONTEXTS):
        baseline_keys, baseline_off = _episode_success(
          _episode_path(
            evaluation_root,
            context=context,
            arm="frozen",
            training_seed=None,
            round_index=0,
            condition="off",
          )
        )
        off_keys, final_off = _episode_success(
          _episode_path(
            evaluation_root,
            context=context,
            arm=arm,
            training_seed=training_seed,
            round_index=4,
            condition="off",
          )
        )
        on_keys, final_on = _episode_success(
          _episode_path(
            evaluation_root,
            context=context,
            arm=arm,
            training_seed=training_seed,
            round_index=4,
            condition="on",
          )
        )
        if baseline_keys != off_keys or off_keys != on_keys:
          raise RuntimeError(
            f"paired episode keys differ for {arm}/{context}/{training_seed}"
          )
        improvement[seed_index, context_index] = final_off - baseline_off
        shield_gap[seed_index, context_index] = final_on - final_off
    arm_result = {
      "cbf_off_improvement": _hierarchical_paired_interval(
        improvement,
        bootstrap_seed=BOOTSTRAP_SEED_BASE + arm_index * 10,
      ),
      "shield_gap": _hierarchical_paired_interval(
        shield_gap,
        bootstrap_seed=BOOTSTRAP_SEED_BASE + arm_index * 10 + 1,
      ),
    }
    results[arm] = arm_result
    for metric, record in arm_result.items():
      csv_rows.append(
        {
          "arm": arm,
          "metric": metric,
          "mean": record["mean"],
          "ci95_lower": record["paired_bootstrap_ci95"][0],
          "ci95_upper": record["paired_bootstrap_ci95"][1],
          "repairs": record["repairs"],
          "regressions": record["regressions"],
          "ties": record["ties"],
          "paired_episode_comparisons": record[
            "paired_episode_comparisons"
          ],
          "bootstrap_samples": record["bootstrap_samples"],
          "bootstrap_seed": record["bootstrap_seed"],
          "confidence_interval_is_gate": False,
        }
      )
  return results, csv_rows


def _training_safety(
  training_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  run_rows: list[dict[str, Any]] = []
  curve_rows: list[dict[str, Any]] = []
  for arm in ADAPTATION_ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        path = training_root / context / arm / f"seed_{seed}" / "training_summary.json"
        if not path.is_file():
          raise FileNotFoundError(path)
        summary = json.loads(path.read_text())
        records = summary["round_metrics"]
        metrics = [record["metrics"] for record in records]
        if len(metrics) != 4:
          raise RuntimeError(f"training run does not contain four rounds: {path}")
        cumulative_transitions = 0
        cumulative_falls = 0
        cumulative_recoveries = 0
        cumulative_nominal_violations = 0
        cumulative_executed_violations = 0
        for record, round_metrics in zip(records, metrics, strict=True):
          transitions = int(round_metrics["training_safety_transition_count"])
          nominal_violations = int(
            round_metrics["training_nominal_violation_count"]
          )
          executed_violations = int(
            round_metrics["training_executed_violation_count"]
          )
          falls = int(round_metrics["rollout_fall_count"])
          recoveries = int(round_metrics["training_shield_recovery_count"])
          cumulative_transitions += transitions
          cumulative_falls += falls
          cumulative_recoveries += recoveries
          cumulative_nominal_violations += nominal_violations
          cumulative_executed_violations += executed_violations
          curve_rows.append(
            {
              "arm": arm,
              "context": context,
              "training_seed": seed,
              "round": int(record["round"]),
              "online_transitions": cumulative_transitions,
              "round_transition_count": transitions,
              "round_falls": falls,
              "cumulative_falls": cumulative_falls,
              "round_shield_recoveries": recoveries,
              "cumulative_shield_recoveries": cumulative_recoveries,
              "round_nominal_violation_count": nominal_violations,
              "round_nominal_violation_fraction": (
                nominal_violations / transitions
              ),
              "cumulative_nominal_violation_fraction": (
                cumulative_nominal_violations / cumulative_transitions
              ),
              "round_executed_violation_count": executed_violations,
              "round_executed_violation_fraction": (
                executed_violations / transitions
              ),
              "cumulative_executed_violation_fraction": (
                cumulative_executed_violations / cumulative_transitions
              ),
              "round_minimum_nominal_barrier_margin": float(
                round_metrics["training_minimum_nominal_barrier_margin"]
              ),
              "round_minimum_executed_barrier_margin": float(
                round_metrics["training_minimum_executed_barrier_margin"]
              ),
            }
          )
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
  return run_rows, curve_rows, arm_summary


def _mean_std(values: list[float]) -> tuple[float, float]:
  return _mean(values), _sample_std(values)


def _write_learning_plots(
  output_dir: Path,
  task_rows: list[dict[str, Any]],
  safety_rows: list[dict[str, Any]],
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = {
    "nominal_ft": "#7f7f7f",
    "reward_only_ft": "#d28e00",
    "filter_only_ft": "#2b6cb0",
    "dual_safe_ft": "#22863a",
  }
  transitions = {
    round_index: round_index * TRANSITIONS_PER_ROUND for round_index in ROUNDS
  }

  task_figure, task_axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
  task_specs = (
    ("cbf_off_success_rate", "CBF-off success", False),
    ("shield_gap", "Shield gap (on − off)", True),
  )
  for axis, (metric, title, zero_line) in zip(
    task_axes, task_specs, strict=True
  ):
    for arm in ADAPTATION_ARMS:
      means: list[float] = []
      deviations: list[float] = []
      for round_index in ROUNDS:
        values = [
          float(row[metric])
          for row in task_rows
          if row["arm"] == arm and row["round"] == round_index
        ]
        mean, deviation = _mean_std(values)
        means.append(mean)
        deviations.append(deviation)
      x = [transitions[round_index] for round_index in ROUNDS]
      lower = [mean - deviation for mean, deviation in zip(means, deviations)]
      upper = [mean + deviation for mean, deviation in zip(means, deviations)]
      axis.plot(
        x,
        means,
        marker="o",
        linewidth=1.8,
        color=colors[arm],
        label=ARM_LABELS[arm],
      )
      axis.fill_between(x, lower, upper, color=colors[arm], alpha=0.12)
    if zero_line:
      axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axis.set_title(title)
    axis.set_xlabel("Online transitions")
    axis.grid(alpha=0.22)
  task_axes[0].set_ylabel("Rate")
  task_axes[0].set_ylim(0.0, 1.0)
  task_axes[1].legend(frameon=False, fontsize=8)
  task_figure.tight_layout()
  for suffix in ("png", "svg"):
    task_figure.savefig(
      output_dir / f"task_learning_curves.{suffix}", dpi=220
    )
  plt.close(task_figure)

  safety_figure, safety_axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
  safety_specs = (
    ("cumulative_falls", "Cumulative training falls"),
    (
      "cumulative_executed_violation_fraction",
      "Executed barrier violation fraction",
    ),
  )
  for axis, (metric, title) in zip(safety_axes, safety_specs, strict=True):
    for arm in ADAPTATION_ARMS:
      x: list[int] = []
      means: list[float] = []
      deviations: list[float] = []
      for round_index in TRAINING_ROUNDS:
        matches = [
          row
          for row in safety_rows
          if row["arm"] == arm and row["round"] == round_index
        ]
        values = [float(row[metric]) for row in matches]
        mean, deviation = _mean_std(values)
        x.append(round_index * TRANSITIONS_PER_ROUND)
        means.append(mean)
        deviations.append(deviation)
      lower = [max(0.0, mean - deviation) for mean, deviation in zip(means, deviations)]
      upper = [mean + deviation for mean, deviation in zip(means, deviations)]
      axis.plot(
        x,
        means,
        marker="o",
        linewidth=1.8,
        color=colors[arm],
        label=ARM_LABELS[arm],
      )
      axis.fill_between(x, lower, upper, color=colors[arm], alpha=0.12)
    axis.set_title(title)
    axis.set_xlabel("Online transitions")
    axis.grid(alpha=0.22)
  safety_axes[0].set_ylabel("Mean per adaptation run")
  safety_axes[1].set_ylabel("Fraction")
  safety_axes[1].legend(frameon=False, fontsize=8)
  safety_figure.tight_layout()
  for suffix in ("png", "svg"):
    safety_figure.savefig(
      output_dir / f"training_safety_curves.{suffix}", dpi=220
    )
  plt.close(safety_figure)


def _task_curve_row(
  *,
  arm: str,
  context: str,
  training_seed: int,
  round_index: int,
  paired: dict[str, Any],
  baseline: dict[str, Any],
) -> dict[str, Any]:
  return {
    "arm": arm,
    "context": context,
    "training_seed": training_seed,
    "round": round_index,
    "online_transitions": round_index * TRANSITIONS_PER_ROUND,
    "cbf_off_success_rate": paired["cbf_off_success_rate"],
    "cbf_on_success_rate": paired["cbf_on_success_rate"],
    "shield_gap": paired["shield_gap"],
    "off_improvement_from_round_0": (
      paired["cbf_off_success_rate"] - baseline["cbf_off_success_rate"]
    ),
    "cbf_off_fall_rate": paired["cbf_off_fall_rate"],
    "cbf_on_fall_rate": paired["cbf_on_fall_rate"],
    "cbf_off_mean_reached_riser": paired["cbf_off_mean_reached_riser"],
    "cbf_on_mean_reached_riser": paired["cbf_on_mean_reached_riser"],
    "cbf_off_mean_return": paired["cbf_off_mean_return"],
    "cbf_on_mean_return": paired["cbf_on_mean_return"],
    "cbf_off_mean_completion_time_s": paired[
      "cbf_off_mean_completion_time_s"
    ],
    "cbf_on_mean_completion_time_s": paired[
      "cbf_on_mean_completion_time_s"
    ],
    "cbf_off_mean_success_completion_time_s": paired[
      "cbf_off_mean_success_completion_time_s"
    ],
    "cbf_on_mean_success_completion_time_s": paired[
      "cbf_on_mean_success_completion_time_s"
    ],
    "cbf_off_nominal_violation_steps_per_riser": paired[
      "cbf_off_nominal_violation_steps_per_riser"
    ],
    "cbf_on_nominal_violation_steps_per_riser": paired[
      "cbf_on_nominal_violation_steps_per_riser"
    ],
    "cbf_off_would_intervene_fraction": paired[
      "cbf_off_would_intervene_fraction"
    ],
    "cbf_on_would_intervene_fraction": paired[
      "cbf_on_would_intervene_fraction"
    ],
    "cbf_off_mean_counterfactual_correction_norm": paired[
      "cbf_off_mean_counterfactual_correction_norm"
    ],
    "cbf_on_mean_correction_norm": paired["cbf_on_mean_correction_norm"],
    "cbf_off_unsafe_overlap_steps_per_riser": paired[
      "cbf_off_unsafe_overlap_steps_per_riser"
    ],
    "cbf_on_unsafe_overlap_steps_per_riser": paired[
      "cbf_on_unsafe_overlap_steps_per_riser"
    ],
    "cbf_off_toe_riser_kick_episode_rate": paired[
      "cbf_off_toe_riser_kick_episode_rate"
    ],
    "cbf_on_toe_riser_kick_episode_rate": paired[
      "cbf_on_toe_riser_kick_episode_rate"
    ],
    "cbf_off_toe_riser_kick_events_per_riser": paired[
      "cbf_off_toe_riser_kick_events_per_riser"
    ],
    "cbf_on_toe_riser_kick_events_per_riser": paired[
      "cbf_on_toe_riser_kick_events_per_riser"
    ],
  }


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
          _task_curve_row(
            arm=arm,
            context=context,
            training_seed=seed,
            round_index=0,
            paired=baseline,
            baseline=baseline,
          )
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
            _task_curve_row(
              arm=arm,
              context=context,
              training_seed=seed,
              round_index=round_index,
              paired=row,
              baseline=baseline,
            )
          )

  training_rows, training_curve_rows, training_by_arm = _training_safety(
    training_root
  )
  paired_statistics, paired_statistics_rows = _paired_statistics(
    evaluation_root
  )
  final_rows = [row for row in curve_rows if row["round"] == 4]
  baseline_off_mean = _mean(
    [float(frozen[context]["cbf_off_success_rate"]) for context in CONTEXTS]
  )
  main_table: list[dict[str, Any]] = [
    {
      "arm": "frozen",
      "training_filter": ARM_FACTORS["frozen"][0],
      "cbf_reward": ARM_FACTORS["frozen"][1],
      "f1_cbf_off_success_rate": frozen["F1"]["cbf_off_success_rate"],
      "f2_cbf_off_success_rate": frozen["F2"]["cbf_off_success_rate"],
      "f3_cbf_off_success_rate": frozen["F3"]["cbf_off_success_rate"],
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
      "mean_cbf_off_would_intervene_fraction": _mean(
        [
          float(frozen[context]["cbf_off_would_intervene_fraction"])
          for context in CONTEXTS
        ]
      ),
      "mean_cbf_off_counterfactual_correction_norm": _mean(
        [
          float(
            frozen[context][
              "cbf_off_mean_counterfactual_correction_norm"
            ]
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
          "f1_cbf_off_success_rate": next(
            float(row["cbf_off_success_rate"])
            for row in rows
            if row["context"] == "F1"
          ),
          "f2_cbf_off_success_rate": next(
            float(row["cbf_off_success_rate"])
            for row in rows
            if row["context"] == "F2"
          ),
          "f3_cbf_off_success_rate": next(
            float(row["cbf_off_success_rate"])
            for row in rows
            if row["context"] == "F3"
          ),
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
          "mean_cbf_off_would_intervene_fraction": _mean(
            [float(row["cbf_off_would_intervene_fraction"]) for row in rows]
          ),
          "mean_cbf_off_counterfactual_correction_norm": _mean(
            [
              float(row["cbf_off_mean_counterfactual_correction_norm"])
              for row in rows
            ]
          ),
        }
      )
    training = training_by_arm[arm]
    table_row = {
      "arm": arm,
      "training_filter": ARM_FACTORS[arm][0],
      "cbf_reward": ARM_FACTORS[arm][1],
      "f1_cbf_off_success_rate": _mean(
        [float(row["f1_cbf_off_success_rate"]) for row in per_seed]
      ),
      "f2_cbf_off_success_rate": _mean(
        [float(row["f2_cbf_off_success_rate"]) for row in per_seed]
      ),
      "f3_cbf_off_success_rate": _mean(
        [float(row["f3_cbf_off_success_rate"]) for row in per_seed]
      ),
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
      "mean_cbf_off_would_intervene_fraction": _mean(
        [
          float(row["mean_cbf_off_would_intervene_fraction"])
          for row in per_seed
        ]
      ),
      "mean_cbf_off_counterfactual_correction_norm": _mean(
        [
          float(row["mean_cbf_off_counterfactual_correction_norm"])
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

  for row in main_table:
    arm = str(row["arm"])
    if arm == "frozen":
      continue
    statistics_record = paired_statistics[arm]
    if not math.isclose(
      float(statistics_record["cbf_off_improvement"]["mean"]),
      float(row["mean_off_improvement"]),
      abs_tol=1.0e-12,
    ):
      raise RuntimeError(f"paired improvement mean differs for {arm}")
    if not math.isclose(
      float(statistics_record["shield_gap"]["mean"]),
      float(row["mean_shield_gap"]),
      abs_tol=1.0e-12,
    ):
      raise RuntimeError(f"paired shield-gap mean differs for {arm}")

  dual = next(row for row in main_table if row["arm"] == "dual_safe_ft")
  best_off = max(float(row["mean_cbf_off_success_rate"]) for row in main_table)
  lowest_violation = min(
    float(row["mean_cbf_off_nominal_violation_steps_per_riser"])
    for row in main_table
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
  filtered_training_arms = ("filter_only_ft", "dual_safe_ft")
  training_safety_checks = {
    "filtered_training_fall_free": all(
      int(seed_row["falls"]) == 0
      for arm in filtered_training_arms
      for seed_row in training_by_arm[arm]["per_seed"]
    ),
    "filtered_training_executed_violation_free": all(
      float(seed_row["executed_violation_fraction"]) == 0.0
      for arm in filtered_training_arms
      for seed_row in training_by_arm[arm]["per_seed"]
    ),
    "filtered_training_reduces_executed_violations_vs_nominal": all(
      float(training_by_arm[arm]["training_executed_violation_fraction_mean"])
      < float(
        training_by_arm["nominal_ft"][
          "training_executed_violation_fraction_mean"
        ]
      )
      for arm in filtered_training_arms
    ),
  }
  arm_rows = {str(row["arm"]): row for row in main_table}
  factorial_metrics = (
    "mean_cbf_off_success_rate",
    "mean_cbf_off_nominal_violation_steps_per_riser",
    "mean_cbf_off_would_intervene_fraction",
    "mean_cbf_off_counterfactual_correction_norm",
    "mean_shield_gap",
    "training_falls_mean",
    "training_shield_recoveries_mean",
    "training_nominal_violation_fraction_mean",
    "training_executed_violation_fraction_mean",
  )
  factorial_contrasts: list[dict[str, Any]] = []
  for metric in factorial_metrics:
    nominal = float(arm_rows["nominal_ft"][metric])
    reward = float(arm_rows["reward_only_ft"][metric])
    filtered = float(arm_rows["filter_only_ft"][metric])
    dual = float(arm_rows["dual_safe_ft"][metric])
    factorial_contrasts.append(
      {
        "metric": metric,
        "reward_effect_without_filter": reward - nominal,
        "reward_effect_with_filter": dual - filtered,
        "filter_effect_without_reward": filtered - nominal,
        "filter_effect_with_reward": dual - reward,
        "filter_reward_interaction": dual - filtered - reward + nominal,
        "report_only": True,
      }
    )
  result = {
    "schema_version": 1,
    "experiment": "paper_filter_free_deployment_ablation",
    "primary_metric": "round_4_cbf_off_success_rate",
    "baseline_mean_cbf_off_success_rate": baseline_off_mean,
    "main_table": main_table,
    "details_by_arm": detailed_by_arm,
    "paired_report_statistics": paired_statistics,
    "main_claim_checks": claim_checks,
    "main_claim_comparison_arms": ["frozen", *ARMS],
    "main_claim_supported": all(claim_checks.values()),
    "training_safety_checks": training_safety_checks,
    "global_fall_and_violation_free_training_supported": (
      training_safety_checks["filtered_training_fall_free"]
      and training_safety_checks[
        "filtered_training_executed_violation_free"
      ]
    ),
    "factorial_contrasts": factorial_contrasts,
    "factorial_contrasts_are_report_only": True,
    "curve_row_count": len(curve_rows),
    "training_safety_run_count": len(training_rows),
    "training_safety_curve_row_count": len(training_curve_rows),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_json(output_dir / "final_results.json", result)
  _write_csv(output_dir / "main_table.csv", main_table)
  _write_csv(output_dir / "learning_curves.csv", curve_rows)
  _write_csv(output_dir / "training_safety.csv", training_rows)
  _write_csv(
    output_dir / "training_safety_curves.csv", training_curve_rows
  )
  _write_learning_plots(output_dir, curve_rows, training_curve_rows)
  _atomic_json(output_dir / "paired_statistics.json", paired_statistics)
  _write_csv(output_dir / "paired_statistics.csv", paired_statistics_rows)
  _write_csv(output_dir / "factorial_contrasts.csv", factorial_contrasts)

  lines = [
    "# Filter-free deployment ablation",
    "",
    "| Arm | Filter | CBF reward | F1 off | F2 off | F3 off | Mean off | Off Δ | Shield gap | Training falls |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in main_table:
    falls = (
      "—"
      if row["training_falls_mean"] is None
      else f"{float(row['training_falls_mean']):.1f}"
    )
    lines.append(
      "| {arm} | {filter} | {reward} | {f1:.3f} | {f2:.3f} | {f3:.3f} | "
      "{off:.3f} | {improvement:+.3f} | {gap:+.3f} | {falls} |".format(
        arm=ARM_LABELS[str(row["arm"])],
        filter=row["training_filter"],
        reward=row["cbf_reward"],
        f1=float(row["f1_cbf_off_success_rate"]),
        f2=float(row["f2_cbf_off_success_rate"]),
        f3=float(row["f3_cbf_off_success_rate"]),
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
      "round-4 CBF-off success and the lowest nominal violation rate across "
      "all five main-table methods, including Frozen.",
      "",
      "Global fall- and executed-violation-free training supported: "
      f"**{result['global_fall_and_violation_free_training_supported']}**.",
      "The filtered arms are reported as local barrier-safety improvements, "
      "not as globally fall-free adaptation, whenever this check is false.",
      "",
      "## CBF-dependence diagnostics",
      "",
      "| Arm | Would intervene | Correction norm | Nominal violation/riser | Shield gap |",
      "|---|---:|---:|---:|---:|",
      *[
        "| {arm} | {intervene:.4f} | {correction:.4f} | {violation:.4f} | "
        "{gap:+.4f} |".format(
          arm=ARM_LABELS[str(row["arm"])],
          intervene=float(row["mean_cbf_off_would_intervene_fraction"]),
          correction=float(
            row["mean_cbf_off_counterfactual_correction_norm"]
          ),
          violation=float(
            row["mean_cbf_off_nominal_violation_steps_per_riser"]
          ),
          gap=float(row["mean_shield_gap"]),
        )
        for row in main_table
      ],
      "",
      "## Report-only paired 95% intervals",
      "",
      "| Arm | CBF-off improvement | 95% CI | Shield gap | 95% CI |",
      "|---|---:|---:|---:|---:|",
      *[
        "| {arm} | {gain:+.3f} | [{gain_lo:+.3f}, {gain_hi:+.3f}] | "
        "{gap:+.3f} | [{gap_lo:+.3f}, {gap_hi:+.3f}] |".format(
          arm=ARM_LABELS[arm],
          gain=float(paired_statistics[arm]["cbf_off_improvement"]["mean"]),
          gain_lo=float(
            paired_statistics[arm]["cbf_off_improvement"][
              "paired_bootstrap_ci95"
            ][0]
          ),
          gain_hi=float(
            paired_statistics[arm]["cbf_off_improvement"][
              "paired_bootstrap_ci95"
            ][1]
          ),
          gap=float(paired_statistics[arm]["shield_gap"]["mean"]),
          gap_lo=float(
            paired_statistics[arm]["shield_gap"]["paired_bootstrap_ci95"][0]
          ),
          gap_hi=float(
            paired_statistics[arm]["shield_gap"]["paired_bootstrap_ci95"][1]
          ),
        )
        for arm in ADAPTATION_ARMS
      ],
      "",
      "Intervals resample adaptation seeds and paired episodes while holding "
      "the three deployment contexts fixed and equally weighted. They are "
      "descriptive and do not alter the pre-registered claim gate.",
      "",
      "## Report-only 2×2 factorial contrasts",
      "",
      "| Metric | Reward effect, filter off | Reward effect, filter on | Filter effect, no reward | Filter effect, reward | Interaction |",
      "|---|---:|---:|---:|---:|---:|",
      *[
        "| {metric} | {reward_off:+.5f} | {reward_on:+.5f} | "
        "{filter_no_reward:+.5f} | {filter_reward:+.5f} | "
        "{interaction:+.5f} |".format(
          metric=str(row["metric"]),
          reward_off=float(row["reward_effect_without_filter"]),
          reward_on=float(row["reward_effect_with_filter"]),
          filter_no_reward=float(row["filter_effect_without_reward"]),
          filter_reward=float(row["filter_effect_with_reward"]),
          interaction=float(row["filter_reward_interaction"]),
        )
        for row in factorial_contrasts
      ],
      "",
      "These contrasts are descriptive consequences of the matched 2×2 "
      "ablation and do not alter the pre-registered claim gate.",
      "",
      "![Task learning curves](task_learning_curves.png)",
      "",
      "![Training safety curves](training_safety_curves.png)",
    ]
  )
  (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
