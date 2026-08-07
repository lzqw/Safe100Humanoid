"""Deterministic machine-readable training tables for specialist v20."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROUND_FIELDS = [
  "specialist",
  "adaptation_seed",
  "round",
  "baseline_target_success",
  "baseline_target_fall",
  "baseline_target_return",
  "baseline_target_mean_reached_riser",
  "current_target_success",
  "current_target_fall",
  "current_target_return",
  "current_target_mean_reached_riser",
  "accepted_target_success",
  "accepted_target_fall",
  "accepted_target_return",
  "accepted_target_mean_reached_riser",
  "target_diagnostic_seed",
  "target_diagnostic_episodes",
  "screen_delta_fraction_0_5",
  "screen_delta_fraction_1_0",
  "screen_delta_fraction_1_5",
  "screen_best_fraction",
  "confirmation_success_delta",
  "confirmation_fall_delta",
  "selected_candidate_fraction",
  "retained_candidate_fraction",
  "target_gate_accepted",
  "d0_gate_passed",
  "d0_rollback",
  "d0_checked_candidate_success",
  "d0_checked_candidate_fall",
  "policy_changed_at_round_end",
  "cumulative_retained_updates",
  "ppo_policy_loss",
  "ppo_value_loss",
  "ppo_entropy",
  "ppo_mean_kl",
  "ppo_max_preupdate_kl",
  "normal_transition_fraction",
  "failure_transition_fraction",
  "success_transition_fraction",
  "normal_advantage_mean_before",
  "normal_advantage_std_before",
  "normal_advantage_mean_after",
  "normal_advantage_std_after",
  "failure_advantage_mean_before",
  "failure_advantage_std_before",
  "failure_advantage_mean_after",
  "failure_advantage_std_after",
  "success_advantage_mean_before",
  "success_advantage_std_before",
  "success_advantage_mean_after",
  "success_advantage_std_after",
  "failure_bank_size",
  "success_pool_size",
  "matched_success_bank_size",
  "exact_pair_count",
  "maximum_marginal_imbalance",
  "bank_transactions_committed",
  "bank_transactions_restored",
  "d0_baseline_success",
  "d0_success",
  "d0_success_delta",
  "d0_fall",
  "cbf_interventions_per_riser",
  "cbf_mean_correction_norm",
  "new_input_column_rms",
  "new_input_column_max_abs",
  "legacy_input_column_max_drift",
  "round_end_actor_sha256",
]

CANDIDATE_FIELDS = [
  "specialist",
  "adaptation_seed",
  "round",
  "fraction",
  "screen_seed",
  "screen_episodes",
  "old_success",
  "candidate_success",
  "screen_success_delta",
  "old_fall",
  "candidate_fall",
  "screen_fall_delta",
  "mean_kl",
  "screen_eligible",
  "screen_best",
  "confirmation_seed",
  "confirmation_success_delta",
  "confirmation_fall_delta",
  "target_gate_accepted",
  "d0_rollback",
  "retained",
]

REPLAY_FIELDS = [
  "specialist",
  "adaptation_seed",
  "round",
  "batch",
  "rollout_seed",
  "normal_start_count",
  "failure_start_count",
  "success_start_count",
  "failure_bank_size",
  "success_pool_size",
  "matched_success_bank_size",
  "exact_pair_count",
  "exact_match_passed",
  "maximum_marginal_imbalance",
  "bank_transaction_attempted",
  "bank_transaction_committed",
  "bank_transaction_restored",
  "usable_preflight_passed",
  "cbf_intervention_fraction",
  "cbf_correction_mean",
]


def _metric(source: dict[str, Any] | None, key: str) -> Any:
  return "" if not source or source.get(key) is None else source[key]


def _fraction_map(record: dict[str, Any]) -> dict[float, dict[str, Any]]:
  variants = record.get("candidate_screening", {}).get("variants", [])
  return {float(item["fraction"]): item for item in variants}


def _accepted_target_eval(record: dict[str, Any]) -> dict[str, Any]:
  confirmation = record.get("candidate_confirmation", {})
  if record.get("policy_changed_at_round_end"):
    return confirmation.get("candidate") or {}
  return (
    confirmation.get("old")
    or record.get("candidate_screening", {}).get("old")
    or {}
  )


def _current_target_eval(record: dict[str, Any]) -> dict[str, Any]:
  return (
    record.get("candidate_confirmation", {}).get("old")
    or record.get("candidate_screening", {}).get("old")
    or {}
  )


def _transaction_counts(collectors: list[dict[str, Any]]) -> tuple[int, int]:
  committed = 0
  restored = 0
  for collector in collectors:
    transaction = collector.get("bank_update_transaction") or {}
    committed += int(transaction.get("committed") is True)
    restored += int(
      transaction.get("attempted") is True
      and transaction.get("committed") is False
      and (transaction.get("restored_preflight") or {}).get("passed") is True
    )
  return committed, restored


def round_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  """Flatten one training summary into round 0 plus eight aligned rows."""
  specialist = summary["specialist_mode"]
  seed = int(summary["seed"])
  baseline_target = summary["baseline_eval"]["DQHMED"]
  baseline_d0 = summary["baseline_eval"]["D0"]
  expansion = summary["actor_observation_expansion"]
  rows: list[dict[str, Any]] = [
    {
      "specialist": specialist,
      "adaptation_seed": seed,
      "round": 0,
      "baseline_target_success": baseline_target["success_rate"],
      "baseline_target_fall": baseline_target["fall_rate"],
      "baseline_target_return": baseline_target["mean_return"],
      "baseline_target_mean_reached_riser": baseline_target[
        "mean_reached_riser"
      ],
      "current_target_success": baseline_target["success_rate"],
      "current_target_fall": baseline_target["fall_rate"],
      "current_target_return": baseline_target["mean_return"],
      "current_target_mean_reached_riser": baseline_target[
        "mean_reached_riser"
      ],
      "accepted_target_success": baseline_target["success_rate"],
      "accepted_target_fall": baseline_target["fall_rate"],
      "accepted_target_return": baseline_target["mean_return"],
      "accepted_target_mean_reached_riser": baseline_target[
        "mean_reached_riser"
      ],
      "target_diagnostic_seed": baseline_target["seeds"][0],
      "target_diagnostic_episodes": baseline_target["num_episodes"],
      "cumulative_retained_updates": 0,
      "d0_baseline_success": baseline_d0["success_rate"],
      "d0_success": baseline_d0["success_rate"],
      "d0_success_delta": 0.0,
      "d0_fall": baseline_d0["fall_rate"],
      "cbf_interventions_per_riser": baseline_target[
        "intervention_per_riser"
      ],
      "cbf_mean_correction_norm": baseline_target["correction_mean"],
      "new_input_column_rms": 0.0,
      "new_input_column_max_abs": expansion[
        "new_first_layer_column_max_abs_before_adaptation"
      ],
      "legacy_input_column_max_drift": 0.0,
      "round_end_actor_sha256": summary["initial_actor_sha256"],
    }
  ]
  for record in summary["rounds"]:
    update = record["full_update_metrics"]
    collectors = update.get("collector_metrics", [])
    current = _current_target_eval(record)
    accepted = _accepted_target_eval(record)
    confirmation = record.get("candidate_confirmation", {})
    confirmation_deltas = confirmation.get("deltas", {})
    variants = _fraction_map(record)
    latest_collector = collectors[-1] if collectors else {}
    pair_audits = [
      item.get("matched_pair_audit") or {} for item in collectors
    ]
    committed, restored = _transaction_counts(collectors)
    failure_fraction = float(update["hard_case_transition_fraction"])
    success_fraction = float(
      update["success_counterexample_transition_fraction"]
    )
    d0_candidate = record["d0_check"]["candidate"]
    d0_accepted = record["d0_check"]["accepted_round_end_actor"]
    adapter = record["round_end_adapter"]
    row: dict[str, Any] = {
      "specialist": specialist,
      "adaptation_seed": seed,
      "round": record["round"],
      "baseline_target_success": baseline_target["success_rate"],
      "baseline_target_fall": baseline_target["fall_rate"],
      "baseline_target_return": baseline_target["mean_return"],
      "baseline_target_mean_reached_riser": baseline_target[
        "mean_reached_riser"
      ],
      "current_target_success": _metric(current, "success_rate"),
      "current_target_fall": _metric(current, "fall_rate"),
      "current_target_return": _metric(current, "mean_return"),
      "current_target_mean_reached_riser": _metric(
        current, "mean_reached_riser"
      ),
      "accepted_target_success": _metric(accepted, "success_rate"),
      "accepted_target_fall": _metric(accepted, "fall_rate"),
      "accepted_target_return": _metric(accepted, "mean_return"),
      "accepted_target_mean_reached_riser": _metric(
        accepted, "mean_reached_riser"
      ),
      "target_diagnostic_seed": confirmation.get(
        "seed", record["candidate_screening"]["seed"]
      ),
      "target_diagnostic_episodes": _metric(accepted, "num_episodes"),
      "screen_delta_fraction_0_5": _metric(
        variants.get(0.5), "screen_success_delta"
      ),
      "screen_delta_fraction_1_0": _metric(
        variants.get(1.0), "screen_success_delta"
      ),
      "screen_delta_fraction_1_5": _metric(
        variants.get(1.5), "screen_success_delta"
      ),
      "screen_best_fraction": record["candidate_screening"][
        "best_fraction"
      ],
      "confirmation_success_delta": _metric(
        confirmation_deltas, "success_delta"
      ),
      "confirmation_fall_delta": _metric(
        confirmation_deltas, "fall_delta"
      ),
      "selected_candidate_fraction": record[
        "selected_candidate_fraction"
      ],
      "retained_candidate_fraction": record[
        "retained_candidate_fraction"
      ],
      "target_gate_accepted": record["target_gate_accepted"],
      "d0_gate_passed": record["d0_check"]["passed"],
      "d0_rollback": record["d0_rollback"],
      "d0_checked_candidate_success": d0_candidate["success_rate"],
      "d0_checked_candidate_fall": d0_candidate["fall_rate"],
      "policy_changed_at_round_end": record[
        "policy_changed_at_round_end"
      ],
      "cumulative_retained_updates": record["accepted_update_count"],
      "ppo_policy_loss": update["surrogate"],
      "ppo_value_loss": update["value"],
      "ppo_entropy": update["entropy"],
      "ppo_mean_kl": update["mean_kl"],
      "ppo_max_preupdate_kl": update[
        "maximum_preupdate_minibatch_kl"
      ],
      "normal_transition_fraction": 1.0
      - failure_fraction
      - success_fraction,
      "failure_transition_fraction": failure_fraction,
      "success_transition_fraction": success_fraction,
      "failure_bank_size": _metric(
        latest_collector, "failure_bank_size_after_rollout"
      ),
      "success_pool_size": _metric(
        latest_collector, "success_pool_size_after_rollout"
      ),
      "matched_success_bank_size": _metric(
        latest_collector, "success_bank_size_after_matching"
      ),
      "exact_pair_count": min(
        (audit.get("pair_count", 0) for audit in pair_audits),
        default=0,
      ),
      "maximum_marginal_imbalance": max(
        (
          audit.get("maximum_marginal_imbalance", 0)
          for audit in pair_audits
        ),
        default=0,
      ),
      "bank_transactions_committed": committed,
      "bank_transactions_restored": restored,
      "d0_baseline_success": baseline_d0["success_rate"],
      "d0_success": d0_accepted["success_rate"],
      "d0_success_delta": d0_accepted["success_rate"]
      - baseline_d0["success_rate"],
      "d0_fall": d0_accepted["fall_rate"],
      "cbf_interventions_per_riser": _metric(
        accepted, "intervention_per_riser"
      ),
      "cbf_mean_correction_norm": _metric(accepted, "correction_mean"),
      "new_input_column_rms": adapter["new_input_column_rms"],
      "new_input_column_max_abs": adapter["new_input_column_max_abs"],
      "legacy_input_column_max_drift": adapter[
        "legacy_input_column_change_from_initial_max_abs"
      ],
      "round_end_actor_sha256": record["round_end_actor_sha256"],
    }
    for group in ("normal", "failure", "success"):
      row[f"{group}_advantage_mean_before"] = update[
        f"v19_{group}_advantage_mean_before"
      ]
      row[f"{group}_advantage_std_before"] = update[
        f"v19_{group}_advantage_std_before"
      ]
      row[f"{group}_advantage_mean_after"] = update[
        f"v19_{group}_advantage_mean_after_normalization"
      ]
      row[f"{group}_advantage_std_after"] = update[
        f"v19_{group}_advantage_std_after_normalization"
      ]
    rows.append(row)
  return rows


def candidate_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for record in summary["rounds"]:
    screening = record["candidate_screening"]
    confirmation = record["candidate_confirmation"]
    deltas = confirmation.get("deltas", {})
    for candidate in screening["variants"]:
      evaluation = candidate["screen_eval"]
      update = candidate["update_metrics"]
      fraction = float(candidate["fraction"])
      rows.append(
        {
          "specialist": summary["specialist_mode"],
          "adaptation_seed": summary["seed"],
          "round": record["round"],
          "fraction": fraction,
          "screen_seed": screening["seed"],
          "screen_episodes": screening["episodes_per_candidate"],
          "old_success": screening["old"]["success_rate"],
          "candidate_success": evaluation["success_rate"],
          "screen_success_delta": candidate["screen_success_delta"],
          "old_fall": screening["old"]["fall_rate"],
          "candidate_fall": evaluation["fall_rate"],
          "screen_fall_delta": candidate["screen_fall_delta"],
          "mean_kl": update["mean_kl"],
          "screen_eligible": candidate["screen_eligible"],
          "screen_best": fraction == screening["best_fraction"],
          "confirmation_seed": confirmation["seed"],
          "confirmation_success_delta": _metric(deltas, "success_delta"),
          "confirmation_fall_delta": _metric(deltas, "fall_delta"),
          "target_gate_accepted": record["target_gate_accepted"],
          "d0_rollback": record["d0_rollback"],
          "retained": fraction == record["retained_candidate_fraction"],
        }
      )
  return rows


def replay_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for record in summary["rounds"]:
    collectors = record["full_update_metrics"].get("collector_metrics", [])
    for index, collector in enumerate(collectors, start=1):
      audit = collector.get("matched_pair_audit") or {}
      transaction = collector.get("bank_update_transaction") or {}
      restored = (
        transaction.get("attempted") is True
        and transaction.get("committed") is False
        and (transaction.get("restored_preflight") or {}).get("passed")
        is True
      )
      rows.append(
        {
          "specialist": summary["specialist_mode"],
          "adaptation_seed": summary["seed"],
          "round": record["round"],
          "batch": index,
          "rollout_seed": collector["rollout_seed"],
          "normal_start_count": collector["normal_start_count"],
          "failure_start_count": collector["failure_start_count"],
          "success_start_count": collector["success_start_count"],
          "failure_bank_size": collector[
            "failure_bank_size_after_rollout"
          ],
          "success_pool_size": collector[
            "success_pool_size_after_rollout"
          ],
          "matched_success_bank_size": collector[
            "success_bank_size_after_matching"
          ],
          "exact_pair_count": audit.get("pair_count"),
          "exact_match_passed": audit.get("exact_match_passed"),
          "maximum_marginal_imbalance": audit.get(
            "maximum_marginal_imbalance"
          ),
          "bank_transaction_attempted": transaction.get("attempted"),
          "bank_transaction_committed": transaction.get("committed"),
          "bank_transaction_restored": restored,
          "usable_preflight_passed": (
            transaction.get("usable_preflight") or {}
          ).get("passed"),
          "cbf_intervention_fraction": collector[
            "cbf_intervention_fraction"
          ],
          "cbf_correction_mean": collector["cbf_correction_mean"],
        }
      )
  return rows


def _write_csv(
  path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(
      handle, fieldnames=fieldnames, extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def write_training_tables(
  summary: dict[str, Any], output_dir: Path
) -> dict[str, int]:
  round_rows = round_metric_rows(summary)
  candidate_rows = candidate_metric_rows(summary)
  replay_rows = replay_metric_rows(summary)
  _write_csv(output_dir / "round_metrics.csv", ROUND_FIELDS, round_rows)
  _write_csv(
    output_dir / "candidate_metrics.csv",
    CANDIDATE_FIELDS,
    candidate_rows,
  )
  _write_csv(output_dir / "replay_metrics.csv", REPLAY_FIELDS, replay_rows)
  return {
    "round_metrics_rows": len(round_rows),
    "candidate_metrics_rows": len(candidate_rows),
    "replay_metrics_rows": len(replay_rows),
  }
