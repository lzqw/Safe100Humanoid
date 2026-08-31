"""Deterministically reconstruct the complete v25 evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from cbf_teacher_v25_protocol import (
    ADAPTATION_SEED,
    ALIGNMENT_COVERAGE_MINIMUM,
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_EPISODES,
    CALIBRATION_GAINS,
    CALIBRATION_OFF_SUCCESS_BOUNDS,
    CALIBRATION_ON_SUCCESS_BOUNDS,
    CALIBRATION_REPEATS,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    FINAL_EPISODES,
    FINAL_REPEATS,
    FINAL_SEED_BASE,
    MAX_ACTOR_EPOCHS,
    MINI_BATCHES,
    MINIMUM_INTERVENTION_REDUCTION,
    MINIMUM_OFF_SUCCESS_DELTA,
    MINIMUM_ON_SUCCESS_DELTA,
    NUM_ENVS,
    POLICY_METHOD,
    PRECALIBRATION_REVISION,
    PROTOCOL_ID,
    ROLLOUT_STEPS,
    ROUNDS,
    SHIELD_RESCUE_MINIMUM,
    SOURCE_FILES,
    V23_FINAL_SHA256,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    V24_FINAL_SHA256,
    V24_PROTOCOL_SHA256,
    V24_RESULT_GIT_TREE,
    calibration_evaluation_seed,
    calibration_gate,
    development_gate,
    final_evaluation_seed,
    fixed_deployment_audit_contract,
    fixed_environment_parameters,
    formal_algorithm_parameters,
    paired_repair_regression_counts,
    validate_v25_calibrated_context,
)
from proximal_v23_io import actor_state_sha256, file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--precalibration-protocol", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--calibration-started", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibration-attempts", type=Path, required=True)
    parser.add_argument("--calibration-paired-csv", type=Path, required=True)
    parser.add_argument("--calibration-all-paired-csv", type=Path, required=True)
    parser.add_argument("--calibration-evidence-verification", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-started", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--final-evaluation-started", type=Path, required=True)
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def checkpoint_actor_sha256(path: Path) -> str:
    """Hash the exact actor state stored in an external checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no actor state: {path}")
    return actor_state_sha256(state)


def committed_file_matches(repo: Path, path: Path, commit: str = "HEAD") -> bool:
    """Return whether a repository input is byte-identical to the named commit."""
    try:
        relative = path.resolve().relative_to(repo.resolve())
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (subprocess.CalledProcessError, ValueError):
        return False
    return committed == path.read_bytes()


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


def rollback_reasons_are_protocol_allowed(rounds: list[dict[str, Any]]) -> bool:
    """Reject outcome/performance rollbacks while accepting declared corruption modes."""
    allowed_fragments = (
        "non-finite",
        "non-positive Gaussian standard deviation",
        "moving forward KL exceeded hard ceiling",
        "v25 action/teacher routing audit failed",
        "behavior Gaussian/log-prob routing audit failed",
        "optimizer corruption",
        "v25 update completed no optimizer steps",
    )
    return all(
        row.get("status") != "hard_rollback"
        or (
            isinstance(row.get("rollback_reason"), str)
            and any(
                fragment in row["rollback_reason"] for fragment in allowed_fragments
            )
        )
        for row in rounds
        if isinstance(row, dict)
    ) and all(isinstance(row, dict) for row in rounds)


def round_actor_hash_chain_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Require explicit actor hashes and a continuous eight-round chain."""
    for index, row in enumerate(rounds):
        if not isinstance(row, dict):
            return False
        start = row.get("round_start_actor_sha256")
        end = row.get("round_end_actor_sha256")
        if not _is_sha256(start):
            return False
        if not _is_sha256(end):
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
        if count == 0.0 and (not _close(loss, 0.0) or with_signal != 0.0):
            return False
        if count > 0.0 and with_signal <= 0.0:
            return False
    return True


def exact_final_identity_schedule(rows: list[dict[str, str]]) -> bool:
    """Bind the sole final audit to all 512 frozen fresh identities."""
    try:
        identities = [
            (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        return False
    expected = [
        (final_evaluation_seed(repeat), environment_id)
        for repeat in range(FINAL_REPEATS)
        for environment_id in range(EVAL_BATCH_SIZE)
    ]
    try:
        pair_indices = [int(row["pair_index"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        len(identities) == FINAL_EPISODES
        and identities == expected
        and pair_indices == list(range(FINAL_EPISODES))
    )


def exact_calibration_identity_schedule(
    rows: list[dict[str, str]], candidate_index: int
) -> bool:
    """Require selected calibration rows in deterministic frozen order."""
    try:
        identities = [
            (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        return False
    expected = [
        (calibration_evaluation_seed(candidate_index, repeat), environment_id)
        for repeat in range(CALIBRATION_REPEATS)
        for environment_id in range(EVAL_BATCH_SIZE)
    ]
    return identities == expected


def calibration_evidence_is_valid(
    rows: list[dict[str, str]],
    verification: dict[str, Any],
    calibration: dict[str, Any],
    context: dict[str, Any],
    *,
    csv_path: Path,
) -> bool:
    """Reconstruct every evaluated calibration candidate from compact evidence."""
    attempts = context.get("calibration", {}).get("attempts")
    candidate_records = verification.get("candidates")
    file_record = verification.get("all_evaluated_paired_csv")
    if not all(
        isinstance(value, list) for value in (attempts, candidate_records)
    ) or not isinstance(file_record, dict):
        return False
    if (
        calibration.get("attempts") != attempts
        or calibration.get("candidate_count_evaluated") != len(attempts)
        or verification.get("schema_version") != 1
        or verification.get("protocol_id") != PROTOCOL_ID
        or verification.get("calibration_status") != calibration.get("status")
        or verification.get("candidate_count_evaluated") != len(attempts)
        or verification.get("all_evaluated_paired_episode_count") != len(rows)
        or verification.get("expected_pairs_per_candidate") != CALIBRATION_EPISODES
        or len(rows) != len(attempts) * CALIBRATION_EPISODES
        or len(candidate_records) != len(attempts)
        or verification.get("ordered_frozen_candidate_prefix") is not True
        or verification.get("first_qualifier_rule_verified") is not True
        or verification.get("all_candidate_gates_reconstructed") is not True
        or verification.get("all_raw_off_on_identities_verified") is not True
        or verification.get("selected_paired_csv_matches_raw_arms") is not True
        or verification.get("raw_input_file_count") != len(attempts) * 16
        or not _is_sha256(verification.get("raw_input_inventory_sha256"))
        or file_record.get("path") != csv_path.name
        or file_record.get("bytes") != csv_path.stat().st_size
        or file_record.get("sha256") != file_sha256(csv_path)
        or verification.get("passed") is not True
    ):
        return False
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            grouped.setdefault(int(row["candidate_index"]), []).append(row)
        except (KeyError, TypeError, ValueError):
            return False
    if sorted(grouped) != list(range(len(attempts))):
        return False
    for index, (attempt, record) in enumerate(
        zip(attempts, candidate_records, strict=True)
    ):
        candidate_rows = grouped[index]
        try:
            identities = [
                (int(row["evaluation_seed"]), int(row["environment_id"]))
                for row in candidate_rows
            ]
            gains_match = all(
                _close(float(row["swing_underresponse_gain"]), CALIBRATION_GAINS[index])
                for row in candidate_rows
            )
            off_success = [_bool(row["off_success"]) for row in candidate_rows]
            on_success = [_bool(row["on_success"]) for row in candidate_rows]
            off_kick = [_bool(row["off_toe_riser_kick"]) for row in candidate_rows]
        except (KeyError, TypeError, ValueError):
            return False
        expected = [
            (calibration_evaluation_seed(index, repeat), environment_id)
            for repeat in range(CALIBRATION_REPEATS)
            for environment_id in range(EVAL_BATCH_SIZE)
        ]
        off_failure_count = sum(not value for value in off_success)
        reconstructed = calibration_gate(
            off_success_count=sum(off_success),
            on_success_count=sum(on_success),
            off_toe_riser_failure_count=sum(
                (not success) and kick
                for success, kick in zip(off_success, off_kick, strict=True)
            ),
            off_failure_count=off_failure_count,
            rescued_count=sum(
                (not off) and on
                for off, on in zip(off_success, on_success, strict=True)
            ),
        )
        if (
            identities != expected
            or not gains_match
            or not reconstructed_fields_match(reconstructed, attempt)
            or record.get("candidate_index") != index
            or not _close(
                float(record.get("swing_underresponse_gain", math.nan)),
                CALIBRATION_GAINS[index],
            )
            or record.get("paired_count") != CALIBRATION_EPISODES
            or record.get("actor_state_sha256") != attempt.get("actor_state_sha256")
            or record.get("gate") != reconstructed
            or record.get("gate_matches_summary") is not True
            or record.get("paired_identities_complete_and_unique") is not True
            or record.get("off_on_initial_state_signatures_match") is not True
        ):
            return False
    return True


def execution_markers_are_valid(
    *,
    protocol: dict[str, Any],
    precalibration: dict[str, Any],
    calibration_started: dict[str, Any],
    training_started: dict[str, Any],
    training_completed: dict[str, Any],
    final_started: dict[str, Any],
    protocol_sha256: str,
    precalibration_sha256: str,
    training_sha256: str,
    base_checkpoint_sha256: str,
    final_checkpoint_sha256: str,
    final_actor_sha256: str,
) -> bool:
    """Bind all prospective execution boundaries and forbid hidden repeats."""
    protocol_reference = training_started.get("protocol")
    return (
        precalibration.get("revision") == PRECALIBRATION_REVISION
        and precalibration.get("prospective_execution", {}).get(
            "v25_simulator_episode_started"
        )
        is False
        and calibration_started
        == {
            "protocol_id": PROTOCOL_ID,
            "precalibration_protocol_sha256": precalibration_sha256,
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "ordered_first_qualifier_rule": True,
        }
        and isinstance(protocol_reference, dict)
        and protocol_reference.get("sha256") == protocol_sha256
        and protocol_reference.get("implementation_commit")
        == protocol.get("implementation_boundary", {}).get("git_commit")
        and training_started.get("adapted_policy_outcomes_observed") is False
        and training_started.get("fresh_adaptation_count") == 1
        and training_completed.get("protocol") == protocol_reference
        and training_completed.get("adapted_policy_outcomes_observed") is True
        and training_completed.get("fresh_adaptation_count") == 1
        and training_completed.get("final_actor_sha256") == final_actor_sha256
        and training_completed.get("training_summary_sha256") == training_sha256
        and final_started
        == {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha256,
            "training_summary_sha256": training_sha256,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "final_checkpoint_sha256": final_checkpoint_sha256,
            "condition_order": ["pi0_off", "pi0_on", "pi8_on", "pi8_off"],
            "fresh_condition_count": FINAL_EPISODES,
        }
    )


def supersession_revision_field_is_valid(
    link: dict[str, Any], history_item: dict[str, Any], expected_revision: int
) -> bool:
    """Accept the sole legacy revision-1 link that predates this field."""
    declared = link.get("supersedes_revision")
    if declared == expected_revision:
        return True
    return (
        expected_revision == 1
        and declared is None
        and history_item.get("file")
        == "results/online/proximal_v25/precalibration_protocol.json"
    )


def precalibration_contract_is_valid(
    repo: Path, path: Path, payload: dict[str, Any]
) -> bool:
    """Verify the complete zero-episode supersession chain through this revision."""
    implementation = payload.get("implementation_boundary")
    supersession = payload.get("supersession")
    if not isinstance(implementation, dict) or not isinstance(supersession, dict):
        return False
    history = supersession.get("verified_protocol_history")
    if not isinstance(history, list):
        return False
    expected_revisions = list(range(PRECALIBRATION_REVISION - 1, 0, -1))
    if [
        item.get("revision") for item in history if isinstance(item, dict)
    ] != expected_revisions:
        return False
    if any(not isinstance(item, dict) for item in history):
        return False
    current_payload = payload
    for expected_revision, item in zip(expected_revisions, history, strict=True):
        link = current_payload.get("supersession")
        if (
            not isinstance(link, dict)
            or not supersession_revision_field_is_valid(
                link, item, expected_revision
            )
            or link.get("supersedes_file") != item.get("file")
            or link.get("supersedes_sha256") != item.get("sha256")
            or link.get("superseded_before_any_v25_simulator_episode") is not True
            or link.get("outcomes_observed_before_revision") is not False
        ):
            return False
        try:
            ancestor = (repo / item["file"]).resolve()
            ancestor.relative_to(repo.resolve())
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not ancestor.is_file()
            or file_sha256(ancestor) != item.get("sha256")
            or not committed_file_matches(repo, ancestor)
        ):
            return False
        try:
            prior = json.loads(ancestor.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        declared_revision = prior.get("revision")
        actual_revision = (
            1
            if declared_revision is None
            and ancestor.name == "precalibration_protocol.json"
            else declared_revision
        )
        if (
            prior.get("protocol_id") != PROTOCOL_ID
            or actual_revision != expected_revision
            or prior.get("status")
            != "prospectively_frozen_before_v25_base_only_paired_calibration"
            or prior.get("prospective_execution", {}).get(
                "v25_simulator_episode_started"
            )
            is not False
        ):
            return False
        current_payload = prior
    if current_payload.get("supersession") is not None:
        return False
    implementation_commit = str(implementation.get("git_commit", ""))
    ancestor_check = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
            cwd=repo,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    grid = payload.get("shift_family", {}).get("candidate_grid")
    expected_grid = [
        {
            "candidate_index": index,
            "swing_underresponse_gain": gain,
            "severity_order": "light_to_severe",
            "evaluation_seeds": [
                calibration_evaluation_seed(index, repeat)
                for repeat in range(CALIBRATION_REPEATS)
            ],
            "paired_filter_arms_share_each_seed": True,
        }
        for index, gain in enumerate(CALIBRATION_GAINS)
    ]
    expected_calibration = {
        "base_policy_only": True,
        "adapted_policy_evaluations_used": False,
        "paired_same_initial_conditions_cbf_off_on": True,
        "episodes_per_candidate": CALIBRATION_EPISODES,
        "batch_size": EVAL_BATCH_SIZE,
        "repeats_per_candidate": CALIBRATION_REPEATS,
        "ordered_light_to_severe": True,
        "select_first_qualifier": True,
        "alignment_coverage_minimum": ALIGNMENT_COVERAGE_MINIMUM,
        "shield_rescue_rate_minimum": SHIELD_RESCUE_MINIMUM,
        "off_success_bounds_inclusive": list(CALIBRATION_OFF_SUCCESS_BOUNDS),
        "on_success_bounds_inclusive": list(CALIBRATION_ON_SUCCESS_BOUNDS),
        "toe_riser_event_definition": (
            "debounced entry of the selected swing toe into exact CBF h<=0 half-space"
        ),
        "outcome_dependent_reselection_forbidden": True,
    }
    return (
        payload.get("schema_version") == 1
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("experiment_name") == EXPERIMENT_NAME
        and payload.get("policy_method") == POLICY_METHOD
        and payload.get("status")
        == "prospectively_frozen_before_v25_base_only_paired_calibration"
        and payload.get("revision") == PRECALIBRATION_REVISION
        and path.name
        == f"precalibration_protocol_revision{PRECALIBRATION_REVISION}.json"
        and supersession.get("revision") == PRECALIBRATION_REVISION
        and supersession.get("supersedes_revision") == PRECALIBRATION_REVISION - 1
        and supersession.get("superseded_before_any_v25_simulator_episode") is True
        and supersession.get("outcomes_observed_before_revision") is False
        and isinstance(supersession.get("reason"), str)
        and bool(supersession["reason"].strip())
        and payload.get("base_checkpoint", {}).get("sha256") == BASE_CHECKPOINT_SHA256
        and payload.get("environment") == fixed_environment_parameters()
        and grid == expected_grid
        and payload.get("calibration") == expected_calibration
        and payload.get("training") == formal_algorithm_parameters()
        and set(implementation.get("source_files", {})) == set(SOURCE_FILES)
        and implementation.get(
            "all_execution_sources_committed_before_first_v25_episode"
        )
        is True
        and payload.get("prospective_execution", {}).get(
            "v25_simulator_episode_started"
        )
        is False
        and payload.get("prospective_execution", {}).get("calibration_started") is False
        and payload.get("prospective_execution", {}).get("adaptation_started") is False
        and payload.get("prospective_execution", {}).get("final_evaluation_started")
        is False
        and payload.get("prospective_execution", {}).get(
            "adapted_policy_outcomes_observed"
        )
        is False
        and payload.get("prospective_execution", {}).get(
            "fresh_adaptation_count_planned"
        )
        == 1
        and payload.get("randomness_preflight", {}).get("passed") is True
        and committed_file_matches(repo, path)
        and ancestor_check
    )


def final_checkpoint_contract_is_valid(path: Path) -> bool:
    """Verify that the supplied pi8 checkpoint is the unconditional round-8 file."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["infos"]["cbf_teacher_v25"]
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return payload.get("iter") == ROUNDS and metadata == {
        "round": ROUNDS,
        "boundary": "final",
        "selection": "fixed final round; no validation or performance selection",
    }


def training_commit_bindings_are_valid(
    repo: Path,
    training: dict[str, Any],
    *,
    protocol_path: Path,
    context_path: Path,
) -> bool:
    """Bind the execution commit to the exact frozen protocol and context."""
    commit = training.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode:
        return False
    return committed_file_matches(
        repo, protocol_path, commit
    ) and committed_file_matches(repo, context_path, commit)


def protocol_input_bindings_are_valid(
    protocol: dict[str, Any],
    *,
    precalibration_sha256: str,
    context_sha256: str,
    calibration_started_sha256: str,
    calibration_summary_sha256: str,
    calibration_attempts_sha256: str,
    calibration_paired_sha256: str,
    calibration_all_paired_sha256: str,
    calibration_verification_sha256: str,
) -> bool:
    """Bind the formal protocol to every selected-calibration input byte stream."""
    implementation = protocol.get("implementation_boundary")
    evidence = protocol.get("calibration_evidence")
    if not isinstance(implementation, dict) or not isinstance(evidence, dict):
        return False
    expected = (
        (implementation.get("precalibration_protocol"), precalibration_sha256),
        (implementation.get("calibrated_context"), context_sha256),
        (
            implementation.get("calibration_execution_started"),
            calibration_started_sha256,
        ),
        (implementation.get("calibration_summary"), calibration_summary_sha256),
        (implementation.get("calibration_attempts"), calibration_attempts_sha256),
        (implementation.get("calibration_paired_episodes"), calibration_paired_sha256),
        (
            implementation.get("calibration_all_evaluated_paired_episodes"),
            calibration_all_paired_sha256,
        ),
        (
            implementation.get("calibration_evidence_verification"),
            calibration_verification_sha256,
        ),
        (evidence, calibration_summary_sha256),
        (evidence.get("execution_started"), calibration_started_sha256),
        (evidence.get("attempts"), calibration_attempts_sha256),
        (evidence.get("paired_episodes"), calibration_paired_sha256),
        (
            evidence.get("all_evaluated_paired_episodes"),
            calibration_all_paired_sha256,
        ),
        (evidence.get("independent_reconstruction"), calibration_verification_sha256),
    )
    return all(
        isinstance(reference, dict) and reference.get("sha256") == digest
        for reference, digest in expected
    )


def formal_protocol_contract_is_valid(
    protocol: dict[str, Any], context: dict[str, Any]
) -> bool:
    """Verify the frozen method, exclusions, final schedule, and report gate."""
    implementation = protocol.get("implementation_boundary")
    context_record = protocol.get("context")
    learning = protocol.get("learning_semantics")
    rollback = protocol.get("rollback")
    excluded = protocol.get("excluded")
    final_policy = protocol.get("final_policy")
    evaluation = protocol.get("evaluation")
    report_gate = protocol.get("development_gate")
    fresh = protocol.get("fresh_execution_seeds")
    prospective = protocol.get("prospective_execution")
    dictionaries = (
        implementation,
        context_record,
        learning,
        rollback,
        excluded,
        final_policy,
        evaluation,
        report_gate,
        fresh,
        prospective,
    )
    if not all(isinstance(value, dict) for value in dictionaries):
        return False
    expected_learning = {
        "single_actor": True,
        "single_privileged_critic": True,
        "runtime_cbf_executes_filtered_action": True,
        "ppo_stores_raw_sampled_action_and_behavior_log_probability": True,
        "teacher_actor_coordinates_prevent_double_plant_scaling": True,
        "teacher_target_stop_gradient": True,
        "teacher_success_gate_fixed_before_training": True,
        "moving_reference_round_start_pi_k": True,
        "one_on_policy_batch_per_round": True,
    }
    expected_excluded = {
        "additional_actor_observations": True,
        "specialist_reward": True,
        "failure_or_success_bank": True,
        "state_restart": True,
        "candidate_line_search": True,
        "performance_gate_or_best_checkpoint": True,
        "multiple_critics_or_risk_head": True,
        "multiple_contexts_or_adaptation_seeds": True,
    }
    expected_rollback_reasons = {
        "non-finite actor/critic/loss/gradient state",
        "moving forward KL above 0.01",
        "raw-action or behavior-Gaussian routing corruption",
        "teacher telemetry/reprojection/swing-selection corruption",
        "actor or critic optimizer-state corruption",
    }
    return (
        protocol.get("schema_version") == 1
        and protocol.get("protocol_id") == PROTOCOL_ID
        and protocol.get("policy_method") == POLICY_METHOD
        and protocol.get("status")
        == "prospectively_frozen_after_base_calibration_before_adaptation"
        and implementation.get(
            "all_sources_and_selected_shift_committed_before_adaptation"
        )
        is True
        and protocol.get("environment") == fixed_environment_parameters()
        and context_record.get("context_id") == CONTEXT_ID
        and context_record.get("family") == CONTEXT_FAMILY
        and context_record.get("selected_candidate_index")
        == context["shift"]["selected_candidate_index"]
        and context_record.get("selected_swing_underresponse_gain")
        == context["shift"]["swing_underresponse_gain"]
        and context_record.get("calibration_evaluation_seeds")
        == context["calibration"]["attempts"][-1]["evaluation_seeds"]
        and context_record.get("base_policy_only_first_qualifier") is True
        and context_record.get("adapted_outcomes_used_for_selection") is False
        and protocol.get("training") == formal_algorithm_parameters()
        and learning == expected_learning
        and set(rollback.get("allowed_reasons", [])) == expected_rollback_reasons
        and rollback.get("performance_rollback_forbidden") is True
        and excluded == expected_excluded
        and final_policy.get("rule") == "round 8 actor, independent of performance"
        and final_policy.get("round_checkpoints_recovery_and_curve_only") is True
        and evaluation
        == {
            "conditions": ["pi0_off", "pi0_on", "pi8_on", "pi8_off"],
            "episodes_per_condition": FINAL_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "seed_base": FINAL_SEED_BASE,
            "same_initial_conditions_all_four_conditions": True,
            "deterministic_policy_mean": True,
        }
        and report_gate
        == {
            "minimum_off_success_delta": MINIMUM_OFF_SUCCESS_DELTA,
            "minimum_on_success_delta": MINIMUM_ON_SUCCESS_DELTA,
            "off_kick_rate_must_strictly_decrease": True,
            "minimum_on_intervention_per_riser_relative_reduction": (
                MINIMUM_INTERVENTION_REDUCTION
            ),
            "point_estimates_only": True,
            "used_for_training_rollback_stopping_or_selection": False,
        }
        and fresh
        == {
            "adaptation_seed": ADAPTATION_SEED,
            "final_seed_base": FINAL_SEED_BASE,
        }
        and prospective
        == {
            "calibration_completed": True,
            "adaptation_started": False,
            "final_evaluation_started": False,
            "adapted_policy_outcomes_observed": False,
            "fresh_adaptation_count_planned": 1,
            "outcome_driven_rerun_forbidden": True,
        }
    )


def training_execution_contract_is_valid(
    training: dict[str, Any],
    protocol: dict[str, Any],
    context: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> bool:
    """Verify single-seed, no-bank fixed-round execution and runtime structure."""
    if not isinstance(training, dict) or not rounds:
        return False
    context_record = training.get("context")
    warm_start = training.get("warm_start")
    audit = training.get("structural_audit")
    protocol_record = training.get("protocol")
    if not all(
        isinstance(value, dict)
        for value in (context_record, warm_start, audit, protocol_record)
    ):
        return False
    algorithm = formal_algorithm_parameters()
    expected_audit = {
        "algorithm_class": True,
        "action_config_class": True,
        "actor_observation_dim": 405,
        "critic_observation_dim": 838,
        "actor_observation_groups": ["actor"],
        "critic_observation_groups": ["actor", "critic", "online_privileged"],
        "deployable_failure_group_absent": True,
        "one_actor": True,
        "one_privileged_critic": True,
        "auxiliary_critics_absent": True,
        "specialist_reward_absent": True,
        "runtime_filter": True,
        "phase_selective_shift": True,
        "actor_critic_optimizers_disjoint": True,
        "log_std_trainable_parameter_count": 0,
        "actor_learning_rate": algorithm["actor_learning_rate"],
        "critic_learning_rate": algorithm["critic_learning_rate"],
        "ppo_clip": algorithm["ppo_clip"],
        "maximum_actor_epochs": algorithm["maximum_actor_epochs"],
        "critic_epochs": algorithm["critic_epochs"],
        "mini_batches": algorithm["mini_batches"],
        "moving_kl_beta": algorithm["moving_kl_beta"],
        "target_kl": algorithm["target_kl"],
        "hard_kl_ceiling": algorithm["hard_kl_ceiling"],
        "teacher_distillation_weight": algorithm["teacher_distillation_weight"],
        "teacher_success_horizon": algorithm["teacher_success_horizon_steps"],
        "teacher_correction_scale": algorithm["teacher_correction_scale"],
    }
    audit_matches = all(
        audit.get(key) == value for key, value in expected_audit.items()
    )
    protocol_validation = protocol_record.get("validation")
    expected_protocol_validation_keys = {
        "protocol_id",
        "method",
        "implementation_is_ancestor",
        "randomness_preflight",
        "base_checkpoint",
        "context_file",
        "context_parameters",
        "algorithm",
        "environment",
        "execution_not_started_at_freeze",
        "all_bound_sources_unchanged",
        "protocol_committed_at_head",
        "context_committed_at_head",
    }
    return (
        training.get("schema_version") == 1
        and training.get("protocol_id") == PROTOCOL_ID
        and training.get("smoke") is False
        and training.get("adaptation_seed") == ADAPTATION_SEED
        and training.get("adaptation_seed_count") == 1
        and training.get("state_restart_count") == 0
        and training.get("failure_or_success_bank_count") == 0
        and context_record.get("selected_candidate_index")
        == context["shift"]["selected_candidate_index"]
        and context_record.get("swing_underresponse_gain")
        == context["shift"]["swing_underresponse_gain"]
        and context_record.get("base_policy_only_first_qualifier") is True
        and context_record.get("reused_without_reselection") is True
        and warm_start.get("actor_observation_dim") == 405
        and warm_start.get("critic_observation_dim") == 838
        and warm_start.get("actor_layout") == "exact-original-interface"
        and warm_start.get("critic_layout") == "exact-original-privileged-interface"
        and warm_start.get("source_optimizer_discarded") is True
        and warm_start.get("source_auxiliary_heads_ignored") is True
        and training.get("initial_actor_sha256")
        == rounds[0].get("round_start_actor_sha256")
        and _is_sha256(protocol_record.get("sha256"))
        and protocol_record.get("implementation_commit")
        == protocol.get("implementation_boundary", {}).get("git_commit")
        and context_record.get("metadata", {}).get("shift")
        == "fixed_phase_selective_swing_leg_underresponse"
        and context_record.get("metadata", {}).get("swing_underresponse_gain")
        == context["shift"]["swing_underresponse_gain"]
        and context_record.get("metadata", {}).get("affected_joints_per_swing_leg")
        == ["hip_pitch_joint", "knee_joint", "ankle_pitch_joint"]
        and context_record.get("metadata", {}).get("stance_leg_scale") == 1.0
        and context_record.get("metadata", {}).get("all_other_action_scales") == 1.0
        and context_record.get("metadata", {}).get("runtime_filter") is True
        and context_record.get("metadata", {}).get("actor_observation_fields_added")
        == 0
        and context_record.get("metadata", {}).get("cbf_geometry_exact") is True
        and context_record.get("metadata", {}).get("fixed_deployment_environment")
        == fixed_deployment_audit_contract()
        and all(
            context_record.get("metadata", {}).get(key) is False
            for key in (
                "terrain_geometry_changed",
                "friction_changed",
                "command_changed",
                "controller_changed",
            )
        )
        and isinstance(protocol_validation, dict)
        and set(protocol_validation) == expected_protocol_validation_keys
        and all(value is True for value in protocol_validation.values())
        and audit_matches
    )


def updated_round_dataflow_is_valid(rounds: list[dict[str, Any]]) -> bool:
    """Bind behavior Gaussian routing and fixed loss terms on every update."""
    algorithm = formal_algorithm_parameters()
    for row in rounds:
        if not isinstance(row, dict):
            return False
        if row.get("status") != "updated":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        try:
            distribution_errors = (
                float(metrics["behavior_reference_distribution_param_max_abs_error"]),
                float(metrics["behavior_current_distribution_param_max_abs_error"]),
            )
            log_prob_errors = (
                float(metrics["behavior_reference_log_prob_max_abs_error"]),
                float(metrics["behavior_current_log_prob_max_abs_error"]),
            )
            moving_beta = float(metrics["moving_kl_beta"])
            target_kl = float(metrics["target_kl"])
            hard_ceiling = float(metrics["hard_kl_ceiling"])
            teacher_weight = float(metrics["teacher_distillation_weight"])
            teacher_horizon = int(metrics["teacher_success_horizon"])
            teacher_scale = float(metrics["teacher_correction_scale"])
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
        values = (
            *distribution_errors,
            *log_prob_errors,
            moving_beta,
            target_kl,
            hard_ceiling,
            teacher_weight,
            teacher_scale,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or min(*distribution_errors, *log_prob_errors) < 0.0
            or max(distribution_errors) > 2.0e-5
            or max(log_prob_errors) > 5.0e-4
            or not _close(moving_beta, algorithm["moving_kl_beta"])
            or not _close(target_kl, algorithm["target_kl"])
            or not _close(hard_ceiling, algorithm["hard_kl_ceiling"])
            or not _close(teacher_weight, algorithm["teacher_distillation_weight"])
            or teacher_horizon != algorithm["teacher_success_horizon_steps"]
            or not _close(teacher_scale, algorithm["teacher_correction_scale"])
            or metrics.get("freeze_log_std") is not True
            or metrics.get("round_reference_index") != row.get("round")
        ):
            return False
    return True


def final_execution_contract_is_valid(
    final: dict[str, Any],
    training: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    base_actor_sha256: str,
    protocol_sha256: str,
    training_sha256: str,
    paired_csv_sha256: str,
) -> bool:
    """Verify the one frozen 512-identity four-condition final audit contract."""
    paired = final.get("paired_evaluation")
    checkpoints = final.get("checkpoints")
    conditions = final.get("conditions")
    if not all(isinstance(value, dict) for value in (paired, checkpoints, conditions)):
        return False
    expected_conditions = ("pi0_off", "pi0_on", "pi8_on", "pi8_off")
    expected_seeds = [final_evaluation_seed(repeat) for repeat in range(FINAL_REPEATS)]
    expected_filter = {
        "pi0_off": False,
        "pi0_on": True,
        "pi8_on": True,
        "pi8_off": False,
    }
    signatures = paired.get("initial_state_signatures")
    for name in expected_conditions:
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            return False
        if (
            condition.get("condition") != name
            or condition.get("policy") != name.split("_")[0]
            or condition.get("runtime_filter") is not expected_filter[name]
            or condition.get("num_episodes") != FINAL_EPISODES
            or condition.get("seeds") != expected_seeds
            or condition.get("initial_state_signatures") != signatures
            or condition.get("actor_state_sha256")
            != checkpoints.get(f"{name.split('_')[0]}_actor_sha256")
            or condition.get("checkpoint_sha256")
            != checkpoints.get(f"{name.split('_')[0]}_sha256")
            or condition.get("deterministic_policy_mean") is not True
            or condition.get("one_initial_episode_per_env") is not True
            or condition.get("original_observation_interface") is not True
            or condition.get("actor_observation_dim") != 405
            or float(condition.get("teacher_reprojection_max_abs_error", math.inf))
            > 1.0e-6
            or condition.get("swing_selection_mismatch_count") != 0
        ):
            return False
    return (
        final.get("schema_version") == 1
        and final.get("protocol_id") == PROTOCOL_ID
        and final.get("final_policy_rule") == "round 8 actor, never best-so-far"
        and paired.get("conditions") == list(expected_conditions)
        and paired.get("conditions_per_arm") == FINAL_EPISODES
        and paired.get("same_initial_conditions_all_four_arms") is True
        and paired.get("deterministic_policy_mean") is True
        and paired.get("original_actor_observation_interface") is True
        and paired.get("confidence_intervals_are_gates") is False
        and isinstance(signatures, list)
        and len(signatures) == FINAL_REPEATS
        and all(isinstance(value, str) and value for value in signatures)
        and exact_final_identity_schedule(rows)
        and checkpoints.get("pi0_sha256") == BASE_CHECKPOINT_SHA256
        and checkpoints.get("pi0_actor_sha256") == base_actor_sha256
        and checkpoints.get("pi8_sha256") == training.get("final_checkpoint_sha256")
        and checkpoints.get("pi8_actor_sha256") == training.get("final_actor_sha256")
        and checkpoints.get("pi0_actor_same_off_on") is True
        and checkpoints.get("pi8_actor_same_off_on") is True
        and final.get("protocol_sha256") == protocol_sha256
        and final.get("training_summary_sha256") == training_sha256
        and final.get("paired_episode_metrics_sha256") == paired_csv_sha256
    )


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    paths = (
        args.base_checkpoint,
        args.final_checkpoint,
        args.precalibration_protocol,
        args.protocol,
        args.context,
        args.calibration_started,
        args.calibration_summary,
        args.calibration_attempts,
        args.calibration_paired_csv,
        args.calibration_all_paired_csv,
        args.calibration_evidence_verification,
        args.training_summary,
        args.training_started,
        args.training_completion,
        args.final_evaluation_started,
        args.final_test,
        args.paired_csv,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    precalibration = json.loads(args.precalibration_protocol.read_text())
    protocol = json.loads(args.protocol.read_text())
    context = validate_v25_calibrated_context(json.loads(args.context.read_text()))
    calibration_started = json.loads(args.calibration_started.read_text())
    calibration = json.loads(args.calibration_summary.read_text())
    attempts = json.loads(args.calibration_attempts.read_text())
    with args.calibration_paired_csv.open(newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
    with args.calibration_all_paired_csv.open(newline="") as handle:
        all_calibration_rows = list(csv.DictReader(handle))
    calibration_verification = json.loads(
        args.calibration_evidence_verification.read_text()
    )
    training = json.loads(args.training_summary.read_text())
    training_started = json.loads(args.training_started.read_text())
    training_completed = json.loads(args.training_completion.read_text())
    final_started = json.loads(args.final_evaluation_started.read_text())
    final = json.loads(args.final_test.read_text())
    with args.paired_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rounds = training.get("rounds", [])
    updated = [row for row in rounds if row.get("status") == "updated"]
    rollbacks = [row for row in rounds if row.get("status") == "hard_rollback"]
    base_checkpoint_sha = file_sha256(args.base_checkpoint)
    final_checkpoint_sha = file_sha256(args.final_checkpoint)
    base_actor_sha = checkpoint_actor_sha256(args.base_checkpoint)
    final_actor_sha = checkpoint_actor_sha256(args.final_checkpoint)
    precalibration_sha = file_sha256(args.precalibration_protocol)
    protocol_sha = file_sha256(args.protocol)
    context_sha = file_sha256(args.context)
    calibration_started_sha = file_sha256(args.calibration_started)
    calibration_summary_sha = file_sha256(args.calibration_summary)
    calibration_attempts_sha = file_sha256(args.calibration_attempts)
    calibration_paired_sha = file_sha256(args.calibration_paired_csv)
    all_calibration_paired_sha = file_sha256(args.calibration_all_paired_csv)
    calibration_verification_sha = file_sha256(args.calibration_evidence_verification)
    training_sha = file_sha256(args.training_summary)
    paired_csv_sha = file_sha256(args.paired_csv)

    condition_success = {}
    condition_kick = {}
    condition_interventions = {}
    condition_counts: dict[str, dict[str, int]] = {}
    for condition in ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
        success_count = sum(_bool(row[f"{condition}_success"]) for row in rows)
        kick_count = sum(_bool(row[f"{condition}_toe_riser_kick"]) for row in rows)
        total_risers = sum(int(row[f"{condition}_max_riser"]) for row in rows)
        total_interventions = sum(
            int(row[f"{condition}_intervention_count"]) for row in rows
        )
        total_would_intervene = sum(
            int(row[f"{condition}_would_intervene_count"]) for row in rows
        )
        condition_success[condition] = success_count / max(1, len(rows))
        condition_kick[condition] = kick_count / max(1, len(rows))
        condition_interventions[condition] = total_interventions / max(1, total_risers)
        condition_counts[condition] = {
            "success_count": success_count,
            "failure_count": len(rows) - success_count,
            "fall_count": sum(_bool(row[f"{condition}_fell"]) for row in rows),
            "timeout_count": sum(_bool(row[f"{condition}_timed_out"]) for row in rows),
            "kick_episode_count": kick_count,
            "toe_riser_failure_count": sum(
                (not _bool(row[f"{condition}_success"]))
                and _bool(row[f"{condition}_toe_riser_kick"])
                for row in rows
            ),
            "total_reached_risers": total_risers,
            "total_intervention_count": total_interventions,
            "total_would_intervene_count": total_would_intervene,
        }
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
    try:
        final_identities = [
            (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        final_identities = []
    base_off_failures = [not _bool(row["pi0_off_success"]) for row in rows]
    base_off_success = [not failure for failure in base_off_failures]
    base_on_success = [_bool(row["pi0_on_success"]) for row in rows]
    final_on_success = [_bool(row["pi8_on_success"]) for row in rows]
    final_off_success = [_bool(row["pi8_off_success"]) for row in rows]
    rescued = sum(
        failure and success
        for failure, success in zip(base_off_failures, base_on_success, strict=True)
    )
    aligned = sum(
        failure and _bool(row["pi0_off_toe_riser_kick"])
        for failure, row in zip(base_off_failures, rows, strict=True)
    )
    paired_changes = {
        "off": paired_repair_regression_counts(base_off_success, final_off_success),
        "on": paired_repair_regression_counts(base_on_success, final_on_success),
    }
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
        "precalibration_contract_and_zero_episode_chain": (
            precalibration_contract_is_valid(
                repo, args.precalibration_protocol, precalibration
            )
        ),
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
        and set(source_checks) == set(SOURCE_FILES)
        and all(source_checks.values()),
        "formal_protocol_committed": committed_file_matches(repo, args.protocol),
        "formal_protocol_inputs_bound": protocol_input_bindings_are_valid(
            protocol,
            precalibration_sha256=precalibration_sha,
            context_sha256=context_sha,
            calibration_started_sha256=calibration_started_sha,
            calibration_summary_sha256=calibration_summary_sha,
            calibration_attempts_sha256=calibration_attempts_sha,
            calibration_paired_sha256=calibration_paired_sha,
            calibration_all_paired_sha256=all_calibration_paired_sha,
            calibration_verification_sha256=calibration_verification_sha,
        ),
        "formal_protocol_contract": formal_protocol_contract_is_valid(
            protocol, context
        ),
        "execution_markers_bound": execution_markers_are_valid(
            protocol=protocol,
            precalibration=precalibration,
            calibration_started=calibration_started,
            training_started=training_started,
            training_completed=training_completed,
            final_started=final_started,
            protocol_sha256=protocol_sha,
            precalibration_sha256=precalibration_sha,
            training_sha256=training_sha,
            base_checkpoint_sha256=base_checkpoint_sha,
            final_checkpoint_sha256=final_checkpoint_sha,
            final_actor_sha256=final_actor_sha,
        )
        and training.get("protocol") == training_started.get("protocol"),
        "base_checkpoint": training.get("base_checkpoint", {}).get("sha256")
        == protocol.get("base_checkpoint", {}).get("sha256")
        == base_checkpoint_sha
        == BASE_CHECKPOINT_SHA256,
        "base_checkpoint_actor_bound": calibration.get("selected_actor_state_sha256")
        == selected_calibration_attempt.get("actor_state_sha256")
        == protocol.get("context", {}).get("base_actor_state_sha256")
        == base_actor_sha,
        "final_checkpoint_bound": training.get("final_checkpoint_sha256")
        == final_checkpoint_sha
        and training.get("final_actor_sha256") == final_actor_sha
        and final_checkpoint_contract_is_valid(args.final_checkpoint),
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
        and calibration.get("status") == "first_qualifying_candidate_frozen"
        and calibration.get("ordered_light_to_severe") is True
        and calibration.get("candidate_count_evaluated")
        == context["shift"]["selected_candidate_index"] + 1,
        "calibration_qualifies": calibration.get("selected_gate", {}).get("qualifies")
        is True,
        "calibration_paired_rows_512": len(calibration_rows) == 512,
        "calibration_identities_unique": len(calibration_identities)
        == len(set(calibration_identities)),
        "calibration_identities_exact_frozen_schedule": sorted(calibration_identities)
        == expected_calibration_identities
        and exact_calibration_identity_schedule(calibration_rows, selected_index),
        "calibration_gate_reconstructed": reconstructed_fields_match(
            reconstructed_calibration_gate, calibration.get("selected_gate", {})
        ),
        "calibration_selected_attempt_bound": calibration.get("selected_gate")
        == selected_calibration_attempt
        and calibration.get("attempts") == context["calibration"]["attempts"]
        and attempts == calibration.get("attempts")
        and calibration.get("precalibration_protocol_sha256")
        == context.get("calibration", {}).get("precalibration_protocol_sha256")
        == precalibration_sha,
        "all_calibration_evidence_reconstructed": calibration_evidence_is_valid(
            all_calibration_rows,
            calibration_verification,
            calibration,
            context,
            csv_path=args.calibration_all_paired_csv,
        ),
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
        "training_execution_contract": training_execution_contract_is_valid(
            training, protocol, context, rounds
        ),
        "training_commit_inputs_bound": training_commit_bindings_are_valid(
            repo,
            training,
            protocol_path=args.protocol,
            context_path=args.context,
        ),
        "round_count_8": len(rounds) == 8,
        "round_numbers_1_to_8": [row.get("round") for row in rounds]
        == list(range(1, 9)),
        "round_status_accounting_valid": round_status_accounting_is_valid(rounds),
        "rollback_reasons_protocol_allowed": rollback_reasons_are_protocol_allowed(
            rounds
        ),
        "round_actor_hash_chain": round_actor_hash_chain_is_valid(rounds),
        "updated_round_kl_below_ceiling": updated_round_kl_is_valid(rounds),
        "updated_round_dataflow_and_loss_contract": updated_round_dataflow_is_valid(
            rounds
        ),
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
        "final_execution_contract": final_execution_contract_is_valid(
            final,
            training,
            rows,
            base_actor_sha256=base_actor_sha,
            protocol_sha256=protocol_sha,
            training_sha256=training_sha,
            paired_csv_sha256=paired_csv_sha,
        ),
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
        "condition_counts_reconstructed": all(
            all(conditions[name].get(key) == value for key, value in counts.items())
            for name, counts in condition_counts.items()
        ),
        "primary_deltas_reconstructed": _close(
            off_delta, final["primary_outcomes"]["internalization_delta"]
        )
        and _close(on_delta, final["primary_outcomes"]["shielded_task_delta"])
        and _close(
            condition_kick["pi8_off"] - condition_kick["pi0_off"],
            final["primary_outcomes"]["off_kick_rate_delta"],
        )
        and _close(
            condition_interventions["pi8_on"] - condition_interventions["pi0_on"],
            final["primary_outcomes"]["on_intervention_per_riser_delta"],
        )
        and _close(
            reconstructed_gate["intervention_per_riser_relative_reduction"],
            final["primary_outcomes"]["on_intervention_per_riser_relative_reduction"],
        ),
        "fresh_rescue_reconstructed": rescued
        == final["fresh_pi0_rescue_audit"]["off_failure_to_on_success_count"]
        and final["fresh_pi0_rescue_audit"]["base_off_failure_count"]
        == sum(base_off_failures)
        and _close(
            rescued / max(1, sum(base_off_failures)),
            final["fresh_pi0_rescue_audit"]["shield_rescue_rate"],
        ),
        "fresh_alignment_reconstructed": aligned
        == int(conditions["pi0_off"]["toe_riser_failure_count"])
        and _close(
            aligned / max(1, sum(base_off_failures)),
            final["fresh_pi0_rescue_audit"]["alignment_coverage"],
        ),
        "development_gate_reconstructed": reconstructed_gate
        == final["development_gate"]
        and final.get("interpretation")
        == (
            "development success"
            if reconstructed_gate["passed"]
            else "development gate not met"
        ),
        "paired_repair_regression_counts_reconstructed": paired_changes
        == final.get("paired_changes"),
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
            "base_checkpoint": base_checkpoint_sha,
            "final_checkpoint": final_checkpoint_sha,
            args.precalibration_protocol.name: precalibration_sha,
            "protocol.json": protocol_sha,
            "context.json": context_sha,
            "calibration_execution_started.json": calibration_started_sha,
            "calibration_summary.json": calibration_summary_sha,
            "attempts.json": calibration_attempts_sha,
            "calibration_selected_paired_episodes.csv": calibration_paired_sha,
            "all_evaluated_paired_episodes.csv": all_calibration_paired_sha,
            "calibration_evidence_verification.json": calibration_verification_sha,
            "formal_execution_started.json": file_sha256(args.training_started),
            "training_summary.json": training_sha,
            "formal_execution_completed.json": file_sha256(args.training_completion),
            "final_evaluation_started.json": file_sha256(args.final_evaluation_started),
            "final_test.json": file_sha256(args.final_test),
            "paired_episode_metrics.csv": paired_csv_sha,
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
