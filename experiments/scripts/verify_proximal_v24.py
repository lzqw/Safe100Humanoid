"""Deterministically reconstruct v24 and immutable v23/v24 boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from proximal_v24_protocol import (
    BASE_CHECKPOINT_SHA256,
    EXPERIMENT_NAME,
    POLICY_METHOD,
    PROTOCOL_ID,
    development_gate,
    formal_algorithm_parameters,
    pure_contact_context_audit,
    validate_v24_calibrated_context,
)

V23_PROTOCOL_SHA256 = "745e888e47d9d33fe87fffa4bbaba618a7e91f37b55ebbf8cb08f578fc1d8f38"
V23_FINAL_SHA256 = "7cbbbfc596e5ad39177c946998055fa460c646730a00385701b595f77cff0148"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--final-test", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--v23-protocol", type=Path, required=True)
    parser.add_argument("--v23-final-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid paired CSV boolean: {value!r}")
    return normalized == "true"


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)


def main() -> None:
    args = _parse_args()
    paths = (
        args.protocol,
        args.context,
        args.training_summary,
        args.final_test,
        args.paired_csv,
        args.v23_protocol,
        args.v23_final_test,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(args.protocol.read_text())
    context = validate_v24_calibrated_context(json.loads(args.context.read_text()))
    training = json.loads(args.training_summary.read_text())
    final = json.loads(args.final_test.read_text())
    v23_final = json.loads(args.v23_final_test.read_text())
    with args.paired_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_rows = [row for row in rows if row["domain"] == "target"]
    d0_rows = [row for row in rows if row["domain"] == "D0"]
    rounds = training.get("rounds", [])
    updated_rounds = [row for row in rounds if row.get("status") == "updated"]
    rollback_rounds = [row for row in rounds if row.get("status") == "hard_rollback"]
    accepted_kl = [float(row["metrics"]["moving_forward_kl"]) for row in updated_rounds]
    target_success_old = [_boolean(row["baseline_success"]) for row in target_rows]
    target_success_new = [_boolean(row["final_success"]) for row in target_rows]
    d0_success_old = [_boolean(row["baseline_success"]) for row in d0_rows]
    d0_success_new = [_boolean(row["final_success"]) for row in d0_rows]
    target_fall_old = [_boolean(row["baseline_fell"]) for row in target_rows]
    target_fall_new = [_boolean(row["final_fell"]) for row in target_rows]
    target_success_delta = (sum(target_success_new) - sum(target_success_old)) / max(
        1, len(target_rows)
    )
    target_fall_delta = (sum(target_fall_new) - sum(target_fall_old)) / max(
        1, len(target_rows)
    )
    d0_success_delta = (sum(d0_success_new) - sum(d0_success_old)) / max(
        1, len(d0_rows)
    )
    reconstructed_gate = development_gate(
        target_success_delta=target_success_delta,
        target_fall_delta=target_fall_delta,
        d0_success_delta=d0_success_delta,
    )
    target_repairs = sum(
        (not old) and new
        for old, new in zip(target_success_old, target_success_new, strict=True)
    )
    target_regressions = sum(
        old and (not new)
        for old, new in zip(target_success_old, target_success_new, strict=True)
    )
    d0_repairs = sum(
        (not old) and new
        for old, new in zip(d0_success_old, d0_success_new, strict=True)
    )
    d0_regressions = sum(
        old and (not new)
        for old, new in zip(d0_success_old, d0_success_new, strict=True)
    )
    required_columns = {
        "baseline_return",
        "final_return",
        "baseline_max_riser",
        "final_max_riser",
        "baseline_mean_slip_signal",
        "final_mean_slip_signal",
        "baseline_mean_contact_mismatch",
        "final_mean_contact_mismatch",
        "baseline_intervention_per_riser",
        "final_intervention_per_riser",
        "baseline_mean_correction_norm",
        "final_mean_correction_norm",
        "baseline_recovery_takeover",
        "final_recovery_takeover",
    }
    checks: dict[str, bool] = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "protocol_frozen_before_adaptation": protocol.get(
            "prospective_execution", {}
        ).get("adapted_policy_outcomes_observed")
        is False,
        "protocol_randomness_preflight": protocol.get("randomness_preflight", {}).get(
            "passed"
        )
        is True,
        "algorithm_exactly_v23": protocol.get("training")
        == training.get("training")
        == formal_algorithm_parameters(),
        "base_checkpoint": training.get("base_checkpoint", {}).get("sha256")
        == BASE_CHECKPOINT_SHA256,
        "context_file_bound": protocol.get("context", {}).get("file_sha256")
        == training.get("context", {}).get("file_sha256")
        == file_sha256(args.context),
        "context_parameters_bound": protocol.get("context", {}).get("parameters_sha256")
        == training.get("context", {}).get("parameters_sha256")
        == context["parameters_sha256"],
        "pure_low_friction_only": pure_contact_context_audit(context)["passed"],
        "base_only_first_qualifier": context["calibration"][
            "adapted_policy_evaluations_used"
        ]
        is False,
        "experiment_identity": training.get("experiment_name") == EXPERIMENT_NAME
        and training.get("policy_method") == POLICY_METHOD,
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
        "updated_round_kl_below_hard_ceiling": bool(accepted_kl)
        and max(accepted_kl) <= 0.01,
        "hard_rollbacks_restore_actor": all(
            row["round_start_actor_sha256"] == row["round_end_actor_sha256"]
            for row in rollback_rounds
        ),
        "no_performance_rollback": training.get("performance_rollbacks") == 0,
        "no_performance_selection": training.get("final_policy_rule")
        == "round 8 actor, never best-so-far"
        and training.get("candidate_screen_or_confirmation_count") == 0,
        "final_actor_is_round_8": bool(rounds)
        and training.get("final_actor_sha256") == rounds[-1]["round_end_actor_sha256"],
        "action_routing_exact": all(
            float(row["metrics"].get("executed_action_routing_max_abs_error", 0.0))
            == 0.0
            for row in rounds
        ),
        "policy_storage_exact": all(
            float(row["metrics"].get("policy_storage_max_abs_error", 0.0)) == 0.0
            for row in rounds
        ),
        "paired_row_count_768": len(rows) == 768,
        "paired_target_row_count_512": len(target_rows) == 512,
        "paired_D0_row_count_256": len(d0_rows) == 256,
        "required_contact_telemetry_present": required_columns.issubset(
            rows[0].keys() if rows else set()
        ),
        "paired_initial_conditions": final.get("paired_evaluation", {}).get(
            "base_and_final_initial_conditions_identical"
        )
        is True,
        "runtime_cbf": final.get("paired_evaluation", {}).get("runtime_cbf") is True,
        "original_actor_interface": final.get("paired_evaluation", {}).get(
            "original_actor_observation_interface"
        )
        is True,
        "target_success_reconstructed": _close(
            target_success_delta, final["target"]["success"]["delta"]
        ),
        "target_fall_reconstructed": _close(
            target_fall_delta, final["target"]["fall"]["delta"]
        ),
        "D0_success_reconstructed": _close(
            d0_success_delta, final["D0"]["success"]["delta"]
        ),
        "repairs_regressions_reconstructed": (
            target_repairs == final["target"]["repairs_regressions"]["repair_count"]
            and target_regressions
            == final["target"]["repairs_regressions"]["regression_count"]
            and d0_repairs == final["D0"]["repairs_regressions"]["repair_count"]
            and d0_regressions == final["D0"]["repairs_regressions"]["regression_count"]
        ),
        "development_gate_reconstructed": reconstructed_gate
        == final["development_gate"],
        "confidence_intervals_report_only": final.get("paired_evaluation", {}).get(
            "confidence_intervals_are_report_only"
        )
        is True,
        "single_independent_contact_result": final.get(
            "independent_contact_context_run"
        )
        is True
        and final.get("joint_lateral_contact_gate") is False,
        "v23_protocol_byte_unchanged": file_sha256(args.v23_protocol)
        == V23_PROTOCOL_SHA256,
        "v23_final_byte_unchanged": file_sha256(args.v23_final_test)
        == V23_FINAL_SHA256,
        "v23_lateral_values_unchanged": (
            v23_final["target"]["success"]["baseline_mean"] == 0.693359375
            and v23_final["target"]["success"]["final_mean"] == 0.689453125
            and v23_final["target"]["success"]["delta"] == -0.00390625
            and v23_final["target"]["fall"]["delta"] == 0.00390625
            and v23_final["target"]["repairs_regressions"]["repair_count"] == 93
            and v23_final["target"]["repairs_regressions"]["regression_count"] == 95
        ),
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "verification_scope": (
            "post-outcome deterministic reconstruction; never a training gate"
        ),
        "input_sha256": {
            "protocol.json": file_sha256(args.protocol),
            "context.json": file_sha256(args.context),
            "training_summary.json": file_sha256(args.training_summary),
            "final_test.json": file_sha256(args.final_test),
            "paired_episode_metrics.csv": file_sha256(args.paired_csv),
            "v23_protocol.json": file_sha256(args.v23_protocol),
            "v23_final_test.json": file_sha256(args.v23_final_test),
        },
        "checks": checks,
        "overall_verification_passed": all(checks.values()),
        "maximum_accepted_moving_forward_kl": max(accepted_kl),
        "hard_rollback_count": len(rollback_rounds),
        "paired_rows": {
            "target": len(target_rows),
            "D0": len(d0_rows),
            "total": len(rows),
        },
        "reconstructed": {
            "target_success_delta": target_success_delta,
            "target_fall_delta": target_fall_delta,
            "D0_success_delta": d0_success_delta,
            "target_repairs": target_repairs,
            "target_regressions": target_regressions,
            "D0_repairs": d0_repairs,
            "D0_regressions": d0_regressions,
            "development_gate": reconstructed_gate,
        },
    }
    if not output["overall_verification_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"v24 verification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
