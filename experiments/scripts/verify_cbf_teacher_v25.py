"""Deterministically reconstruct the complete v25 evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v25_protocol import (
    BASE_CHECKPOINT_SHA256,
    FINAL_EPISODES,
    POLICY_METHOD,
    PROTOCOL_ID,
    V23_FINAL_SHA256,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    V24_FINAL_SHA256,
    V24_PROTOCOL_SHA256,
    V24_RESULT_GIT_TREE,
    development_gate,
    formal_algorithm_parameters,
    validate_v25_calibrated_context,
)
from proximal_v23_io import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--final-test", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid paired CSV boolean: {value!r}")
    return normalized == "true"


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    paths = (
        args.protocol,
        args.context,
        args.calibration_summary,
        args.training_summary,
        args.final_test,
        args.paired_csv,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(args.protocol.read_text())
    context = validate_v25_calibrated_context(json.loads(args.context.read_text()))
    calibration = json.loads(args.calibration_summary.read_text())
    training = json.loads(args.training_summary.read_text())
    final = json.loads(args.final_test.read_text())
    with args.paired_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rounds = training.get("rounds", [])
    updated = [row for row in rounds if row.get("status") == "updated"]
    rollbacks = [row for row in rounds if row.get("status") == "hard_rollback"]

    condition_success = {}
    condition_kick = {}
    condition_interventions = {}
    for condition in ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
        condition_success[condition] = sum(
            _bool(row[f"{condition}_success"]) for row in rows
        ) / max(1, len(rows))
        condition_kick[condition] = sum(
            _bool(row[f"{condition}_toe_riser_kick"]) for row in rows
        ) / max(1, len(rows))
        condition_interventions[condition] = sum(
            float(row[f"{condition}_intervention_per_riser"]) for row in rows
        ) / max(1, len(rows))
    off_delta = condition_success["pi8_off"] - condition_success["pi0_off"]
    on_delta = condition_success["pi8_on"] - condition_success["pi0_on"]
    reconstructed_gate = development_gate(
        off_success_delta=off_delta,
        on_success_delta=on_delta,
        base_off_kick_rate=condition_kick["pi0_off"],
        final_off_kick_rate=condition_kick["pi8_off"],
        base_on_intervention_per_riser=condition_interventions["pi0_on"],
        final_on_intervention_per_riser=condition_interventions["pi8_on"],
    )
    base_off_failures = [not _bool(row["pi0_off_success"]) for row in rows]
    base_on_success = [_bool(row["pi0_on_success"]) for row in rows]
    rescued = sum(
        failure and success
        for failure, success in zip(base_off_failures, base_on_success, strict=True)
    )
    aligned = sum(
        failure and _bool(row["pi0_off_toe_riser_kick"])
        for failure, row in zip(base_off_failures, rows, strict=True)
    )
    source_hashes = protocol.get("implementation_boundary", {}).get("source_files", {})
    source_checks = {
        relative: (repo / relative).is_file()
        and file_sha256(repo / relative) == expected
        for relative, expected in source_hashes.items()
    }
    conditions = final["conditions"]
    checks: dict[str, bool] = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "policy_method": protocol.get("policy_method")
        == training.get("policy_method")
        == final.get("policy_method")
        == POLICY_METHOD,
        "protocol_frozen_before_adaptation": protocol.get(
            "prospective_execution", {}
        ).get("adapted_policy_outcomes_observed")
        is False,
        "randomness_preflight": protocol.get("randomness_preflight", {}).get("passed")
        is True,
        "all_bound_source_hashes_current": bool(source_checks)
        and all(source_checks.values()),
        "base_checkpoint": training.get("base_checkpoint", {}).get("sha256")
        == BASE_CHECKPOINT_SHA256,
        "context_file_bound": protocol.get("context", {}).get("file_sha256")
        == training.get("context", {}).get("file_sha256")
        == final.get("context", {}).get("file_sha256")
        == file_sha256(args.context),
        "context_parameters_bound": protocol.get("context", {}).get("parameters_sha256")
        == training.get("context", {}).get("parameters_sha256")
        == final.get("context", {}).get("parameters_sha256")
        == context["parameters_sha256"],
        "base_only_first_qualifier": calibration.get("base_policy_only") is True
        and calibration.get("adapted_policy_evaluations_used") is False
        and calibration.get("candidate_count_evaluated")
        == context["shift"]["selected_candidate_index"] + 1,
        "calibration_qualifies": calibration.get("selected_gate", {}).get("qualifies")
        is True,
        "algorithm_exact": protocol.get("training")
        == training.get("training")
        == formal_algorithm_parameters(),
        "round_count_8": len(rounds) == 8,
        "round_numbers_1_to_8": [row.get("round") for row in rounds]
        == list(range(1, 9)),
        "round_statuses_valid": all(
            row.get("status") in ("updated", "hard_rollback") for row in rounds
        ),
        "round_actor_hash_chain": all(
            rounds[index]["round_start_actor_sha256"]
            == rounds[index - 1]["round_end_actor_sha256"]
            for index in range(1, len(rounds))
        ),
        "updated_round_kl_below_ceiling": bool(updated)
        and all(float(row["metrics"]["moving_forward_kl"]) <= 0.01 for row in updated),
        "hard_rollbacks_restore_actor": all(
            row["round_start_actor_sha256"] == row["round_end_actor_sha256"]
            for row in rollbacks
        ),
        "raw_policy_storage_exact": all(
            float(row["metrics"].get("policy_storage_max_abs_error", 0.0)) <= 1.0e-6
            for row in rounds
        ),
        "runtime_action_routing_exact": all(
            float(row["metrics"].get("executed_action_routing_max_abs_error", 0.0))
            <= 1.0e-5
            for row in rounds
        ),
        "teacher_reprojection_exact": all(
            float(row["metrics"].get("teacher_reprojection_max_abs_error", 0.0))
            <= 1.0e-6
            for row in rounds
        ),
        "teacher_swing_selection_exact": all(
            float(row["metrics"].get("swing_selection_mismatch_count", 0.0)) == 0.0
            for row in rounds
        ),
        "teacher_signal_observed": sum(
            float(row["metrics"].get("teacher_transition_count", 0.0))
            for row in updated
        )
        > 0.0,
        "no_performance_rollback": training.get("performance_rollbacks") == 0,
        "no_performance_selection": training.get("final_policy_rule")
        == "round 8 actor, never best-so-far"
        and training.get("candidate_screen_or_confirmation_count") == 0,
        "final_actor_is_round_8": bool(rounds)
        and training.get("final_actor_sha256") == rounds[-1]["round_end_actor_sha256"],
        "four_condition_row_count_512": len(rows) == FINAL_EPISODES,
        "same_initial_conditions": final.get("paired_evaluation", {}).get(
            "same_initial_conditions_all_four_arms"
        )
        is True,
        "actor_pairs_match": final.get("checkpoints", {}).get("pi0_actor_same_off_on")
        is True
        and final.get("checkpoints", {}).get("pi8_actor_same_off_on") is True,
        "condition_success_reconstructed": all(
            _close(condition_success[name], float(conditions[name]["success_rate"]))
            for name in condition_success
        ),
        "condition_kick_reconstructed": all(
            _close(condition_kick[name], float(conditions[name]["kick_rate"]))
            for name in condition_kick
        ),
        "condition_interventions_reconstructed": all(
            _close(
                condition_interventions[name],
                float(conditions[name]["intervention_per_riser"]),
            )
            for name in condition_interventions
        ),
        "primary_deltas_reconstructed": _close(
            off_delta, final["primary_outcomes"]["internalization_delta"]
        )
        and _close(on_delta, final["primary_outcomes"]["shielded_task_delta"]),
        "fresh_rescue_reconstructed": rescued
        == final["fresh_pi0_rescue_audit"]["off_failure_to_on_success_count"],
        "fresh_alignment_reconstructed": aligned
        == int(conditions["pi0_off"]["toe_riser_failure_count"])
        and _close(
            aligned / max(1, sum(base_off_failures)),
            final["fresh_pi0_rescue_audit"]["alignment_coverage"],
        ),
        "development_gate_reconstructed": reconstructed_gate
        == final["development_gate"],
        "v23_protocol_byte_unchanged": file_sha256(
            repo / "results/online/proximal_v23/protocol.json"
        )
        == V23_PROTOCOL_SHA256,
        "v23_final_byte_unchanged": file_sha256(
            repo / "results/online/proximal_v23/final/final_test.json"
        )
        == V23_FINAL_SHA256,
        "v23_tree_unchanged": _git_output(
            repo, "rev-parse", "HEAD:results/online/proximal_v23"
        )
        == V23_RESULT_GIT_TREE,
        "v24_protocol_byte_unchanged": file_sha256(
            repo / "results/online/proximal_v24/protocol.json"
        )
        == V24_PROTOCOL_SHA256,
        "v24_final_byte_unchanged": file_sha256(
            repo / "results/online/proximal_v24/final/final_test.json"
        )
        == V24_FINAL_SHA256,
        "v24_tree_unchanged": _git_output(
            repo, "rev-parse", "HEAD:results/online/proximal_v24"
        )
        == V24_RESULT_GIT_TREE,
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "verification_scope": "post-outcome deterministic reconstruction; never a training gate",
        "input_sha256": {
            "protocol.json": file_sha256(args.protocol),
            "context.json": file_sha256(args.context),
            "calibration_summary.json": file_sha256(args.calibration_summary),
            "training_summary.json": file_sha256(args.training_summary),
            "final_test.json": file_sha256(args.final_test),
            "paired_episode_metrics.csv": file_sha256(args.paired_csv),
        },
        "source_checks": source_checks,
        "checks": checks,
        "passed": all(checks.values()),
        "check_count": len(checks),
        "failed_checks": [name for name, value in checks.items() if not value],
        "reconstructed": {
            "condition_success": condition_success,
            "condition_kick": condition_kick,
            "condition_intervention_per_riser": condition_interventions,
            "off_success_delta": off_delta,
            "on_success_delta": on_delta,
            "fresh_rescue_count": rescued,
            "fresh_aligned_failure_count": aligned,
            "development_gate": reconstructed_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
