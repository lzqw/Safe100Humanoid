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
    CALIBRATION_REPEATS,
    EVAL_BATCH_SIZE,
    FINAL_EPISODES,
    MAX_ACTOR_EPOCHS,
    MINI_BATCHES,
    NUM_ENVS,
    POLICY_METHOD,
    PROTOCOL_ID,
    ROLLOUT_STEPS,
    V23_FINAL_SHA256,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    V24_FINAL_SHA256,
    V24_PROTOCOL_SHA256,
    V24_RESULT_GIT_TREE,
    calibration_evaluation_seed,
    calibration_gate,
    development_gate,
    fixed_environment_parameters,
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
    parser.add_argument("--calibration-paired-csv", type=Path, required=True)
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


def reconstructed_fields_match(
    reconstructed: dict[str, Any], recorded: dict[str, Any]
) -> bool:
    """Compare reconstructed fields without rejecting bound metadata fields."""
    if not isinstance(recorded, dict):
        return False
    for key, value in reconstructed.items():
        candidate = recorded.get(key)
        if isinstance(value, float):
            if not isinstance(candidate, (int, float)) or not _close(
                value, float(candidate)
            ):
                return False
        elif candidate != value:
            return False
    return True


def updated_metric_is_bounded(
    rounds: list[dict[str, Any]],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    """Require a finite bounded metric on every committed update."""
    for row in rounds:
        if not isinstance(row, dict):
            return False
        if row.get("status") != "updated":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        try:
            value = float(metrics[key])
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
    return True


def round_status_accounting_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Bind update/rollback metadata without requiring either outcome."""
    for row in rounds:
        if not isinstance(row, dict):
            return False
        status = row.get("status")
        reason = row.get("rollback_reason")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        if (
            row.get("round_reference_is_moving_pi_k") is not True
            or row.get("performance_evaluation_or_gate_used") is not False
        ):
            return False
        if status == "updated":
            if reason is not None or metrics.get("hard_rollback") is True:
                return False
        elif status == "hard_rollback":
            if not isinstance(reason, str) or not reason.strip():
                return False
            if metrics.get("hard_rollback") is not True:
                return False
            if metrics.get("hard_rollback_reason") != reason:
                return False
            start = row.get("round_start_actor_sha256")
            end = row.get("round_end_actor_sha256")
            if not isinstance(start, str) or not start or start != end:
                return False
        else:
            return False
    return True


def round_actor_hash_chain_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Require explicit actor hashes and a continuous eight-round chain."""
    for index, row in enumerate(rounds):
        if not isinstance(row, dict):
            return False
        start = row.get("round_start_actor_sha256")
        end = row.get("round_end_actor_sha256")
        if not isinstance(start, str) or not start:
            return False
        if not isinstance(end, str) or not end:
            return False
        if index and start != rounds[index - 1].get("round_end_actor_sha256"):
            return False
    return True


def updated_round_kl_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Validate every update that occurred; an all-rollback run remains auditable."""
    for row in rounds:
        if not isinstance(row, dict):
            return False
        if row.get("status") != "updated":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        try:
            value = float(metrics["moving_forward_kl"])
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value > 0.01:
            return False
    return True


def teacher_signal_accounting_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Validate teacher metrics without requiring an outcome-dependent signal."""
    for row in rounds:
        if not isinstance(row, dict):
            return False
        if row.get("status") != "updated":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        try:
            count = float(metrics["teacher_transition_count"])
            fraction = float(metrics["teacher_transition_fraction"])
            loss = float(metrics["teacher_loss"])
            with_signal = float(metrics["teacher_minibatches_with_signal"])
            without_signal = float(metrics["teacher_minibatches_without_signal"])
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
        values = (count, fraction, loss, with_signal, without_signal)
        if not all(math.isfinite(value) for value in values):
            return False
        if count < 0.0 or not 0.0 <= fraction <= 1.0:
            return False
        if min(loss, with_signal, without_signal) < 0.0:
            return False
        if not all(
            value.is_integer() for value in (count, with_signal, without_signal)
        ):
            return False
        if not _close(fraction, count / (NUM_ENVS * ROLLOUT_STEPS)):
            return False
        try:
            actor_epochs = float(metrics["actor_epochs_completed"])
            actor_minibatches = float(metrics["actor_minibatches_completed"])
            samples_seen = float(metrics["teacher_samples_seen_across_epochs"])
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
        if not all(
            math.isfinite(value)
            for value in (actor_epochs, actor_minibatches, samples_seen)
        ):
            return False
        if not all(
            value >= 0.0 and value.is_integer()
            for value in (actor_epochs, actor_minibatches, samples_seen)
        ):
            return False
        if with_signal + without_signal != actor_minibatches:
            return False
        if not 1.0 <= actor_epochs <= MAX_ACTOR_EPOCHS:
            return False
        if actor_minibatches != actor_epochs * MINI_BATCHES:
            return False
        if samples_seen != count * actor_epochs:
            return False
        if count == 0.0 and (
            not _close(loss, 0.0) or with_signal != 0.0
        ):
            return False
        if count > 0.0 and with_signal <= 0.0:
            return False
    return True


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    paths = (
        args.protocol,
        args.context,
        args.calibration_summary,
        args.calibration_paired_csv,
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
    with args.calibration_paired_csv.open(newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
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
        total_risers = sum(int(row[f"{condition}_max_riser"]) for row in rows)
        total_interventions = sum(
            int(row[f"{condition}_intervention_count"]) for row in rows
        )
        condition_interventions[condition] = total_interventions / max(1, total_risers)
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
    final_identities = [
        (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
    ]
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
    selected_index = int(calibration["selected_candidate_index"])
    calibration_identities = [
        (int(row["evaluation_seed"]), int(row["environment_id"]))
        for row in calibration_rows
    ]
    expected_calibration_identities = [
        (calibration_evaluation_seed(selected_index, repeat), environment_id)
        for repeat in range(CALIBRATION_REPEATS)
        for environment_id in range(EVAL_BATCH_SIZE)
    ]
    calibration_off_success = [_bool(row["off_success"]) for row in calibration_rows]
    calibration_on_success = [_bool(row["on_success"]) for row in calibration_rows]
    calibration_off_kick = [
        _bool(row["off_toe_riser_kick"]) for row in calibration_rows
    ]
    calibration_failure_count = sum(not value for value in calibration_off_success)
    reconstructed_calibration_gate = calibration_gate(
        off_success_count=sum(calibration_off_success),
        on_success_count=sum(calibration_on_success),
        off_toe_riser_failure_count=sum(
            (not success) and kick
            for success, kick in zip(
                calibration_off_success, calibration_off_kick, strict=True
            )
        ),
        off_failure_count=calibration_failure_count,
        rescued_count=sum(
            (not off) and on
            for off, on in zip(
                calibration_off_success, calibration_on_success, strict=True
            )
        ),
    )
    selected_calibration_attempt = context["calibration"]["attempts"][-1]
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
        "calibration_paired_rows_512": len(calibration_rows) == 512,
        "calibration_identities_unique": len(calibration_identities)
        == len(set(calibration_identities)),
        "calibration_identities_exact_frozen_schedule": sorted(calibration_identities)
        == expected_calibration_identities,
        "calibration_gate_reconstructed": reconstructed_fields_match(
            reconstructed_calibration_gate, calibration.get("selected_gate", {})
        ),
        "calibration_selected_attempt_bound": calibration.get("selected_gate")
        == selected_calibration_attempt,
        "calibration_paired_csv_bound": protocol.get("calibration_evidence", {})
        .get("paired_episodes", {})
        .get("sha256")
        == file_sha256(args.calibration_paired_csv)
        == calibration.get("selected_paired_csv_sha256"),
        "algorithm_exact": protocol.get("training")
        == training.get("training")
        == formal_algorithm_parameters(),
        "fixed_environment_exact": protocol.get("environment")
        == training.get("environment")
        == fixed_environment_parameters(),
        "round_count_8": len(rounds) == 8,
        "round_numbers_1_to_8": [row.get("round") for row in rounds]
        == list(range(1, 9)),
        "round_status_accounting_valid": round_status_accounting_is_valid(rounds),
        "round_actor_hash_chain": round_actor_hash_chain_is_valid(rounds),
        "updated_round_kl_below_ceiling": updated_round_kl_is_valid(rounds),
        "raw_policy_storage_exact": updated_metric_is_bounded(
            rounds,
            "policy_storage_max_abs_error",
            minimum=0.0,
            maximum=1.0e-6,
        ),
        "runtime_action_routing_exact": updated_metric_is_bounded(
            rounds,
            "executed_action_routing_max_abs_error",
            minimum=0.0,
            maximum=1.0e-5,
        ),
        "teacher_reprojection_exact": updated_metric_is_bounded(
            rounds,
            "teacher_reprojection_max_abs_error",
            minimum=0.0,
            maximum=1.0e-6,
        ),
        "teacher_swing_selection_exact": updated_metric_is_bounded(
            rounds,
            "swing_selection_mismatch_count",
            minimum=0.0,
            maximum=0.0,
        ),
        "teacher_signal_accounting_valid": teacher_signal_accounting_is_valid(rounds),
        "hard_rollback_count_reconstructed": training.get("hard_rollback_count")
        == len(rollbacks),
        "no_performance_rollback": training.get("performance_rollbacks") == 0,
        "no_performance_selection": training.get("final_policy_rule")
        == "round 8 actor, never best-so-far"
        and training.get("candidate_screen_or_confirmation_count") == 0,
        "final_actor_is_round_8": bool(rounds)
        and training.get("final_actor_sha256")
        == rounds[-1].get("round_end_actor_sha256"),
        "four_condition_row_count_512": len(rows) == FINAL_EPISODES,
        "four_condition_identities_unique": len(final_identities)
        == len(set(final_identities)),
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
            "calibration_selected_paired_episodes.csv": file_sha256(
                args.calibration_paired_csv
            ),
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
            "teacher_transition_count_total": sum(
                float(row.get("metrics", {}).get("teacher_transition_count", 0.0))
                for row in updated
            ),
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
