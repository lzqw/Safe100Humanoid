"""Freeze selected v25 shift before the sole eight-round adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v25_protocol import (
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    FINAL_EPISODES,
    FINAL_SEED_BASE,
    MINIMUM_INTERVENTION_REDUCTION,
    MINIMUM_OFF_SUCCESS_DELTA,
    MINIMUM_ON_SUCCESS_DELTA,
    POLICY_METHOD,
    PROTOCOL_ID,
    formal_algorithm_parameters,
    fresh_randomness_report,
    validate_v25_calibrated_context,
)
from freeze_cbf_teacher_v25_precalibration import (
    _prior_immutable_audit,
    _verify_committed_sources,
)
from proximal_v23_io import file_sha256


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_file(repo: Path, path: Path, commit: str) -> dict[str, str]:
    relative = path.relative_to(repo)
    content = path.read_bytes()
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if content != blob:
        raise RuntimeError(f"formal v25 input is not committed: {relative}")
    return {
        "file": str(relative),
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_commit": commit,
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise RuntimeError(f"refusing to overwrite another v25 protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--precalibration-protocol", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    pre_path = args.precalibration_protocol.resolve()
    context_path = args.context.resolve()
    calibration_path = args.calibration_summary.resolve()
    formal_output_dir = args.formal_output_dir.resolve()
    output = args.output.resolve()
    for path in (checkpoint, pre_path, context_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before formal v25 freeze")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("formal v25 checkpoint differs from frozen pi0")
    if (formal_output_dir / "formal_execution_started.json").exists():
        raise RuntimeError("v25 adaptation started before formal freeze")
    if (repo / "results/online/proximal_v25/final/final_test.json").exists():
        raise RuntimeError("v25 final outcome exists before formal freeze")

    source_hashes = _verify_committed_sources(repo, commit)
    pre_ref = _committed_file(repo, pre_path, commit)
    context_ref = _committed_file(repo, context_path, commit)
    calibration_ref = _committed_file(repo, calibration_path, commit)
    pre = json.loads(pre_path.read_text())
    context = validate_v25_calibrated_context(json.loads(context_path.read_text()))
    calibration = json.loads(calibration_path.read_text())
    selected = context["calibration"]["attempts"][-1]
    pre_source_hashes = pre.get("implementation_boundary", {}).get("source_files", {})
    calibration_checks = {
        "precalibration_id": pre.get("protocol_id") == PROTOCOL_ID,
        "precalibration_status": pre.get("status")
        == "prospectively_frozen_before_v25_base_only_paired_calibration",
        "implementation_unchanged_since_precalibration": pre_source_hashes
        == source_hashes,
        "base_only": calibration.get("base_policy_only") is True,
        "adapted_policy_absent": calibration.get("adapted_policy_evaluations_used")
        is False,
        "first_qualifier": calibration.get("candidate_count_evaluated")
        == len(context["calibration"]["attempts"]),
        "context_file": calibration.get("frozen_context_file_sha256")
        == file_sha256(context_path),
        "parameters": calibration.get("parameters_sha256")
        == context["parameters_sha256"],
        "selected_index": calibration.get("selected_candidate_index")
        == selected["candidate_index"],
        "selected_gain": calibration.get("selected_swing_underresponse_gain")
        == selected["swing_underresponse_gain"],
        "qualifies": selected.get("qualifies") is True,
        "precalibration_sha": context["calibration"].get(
            "precalibration_protocol_sha256"
        )
        == file_sha256(pre_path),
    }
    if not all(calibration_checks.values()):
        raise RuntimeError(f"formal v25 calibration mismatch: {calibration_checks}")
    randomness = fresh_randomness_report(repo)
    if not randomness["passed"]:
        raise RuntimeError(
            f"v25 fresh randomness collision: {randomness['collisions']}"
        )
    prior_audit = _prior_immutable_audit(repo, commit)

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": "prospectively_frozen_after_base_calibration_before_adaptation",
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
            "precalibration_protocol": pre_ref,
            "calibrated_context": context_ref,
            "calibration_summary": calibration_ref,
            "all_sources_and_selected_shift_committed_before_adaptation": True,
        },
        "prior_results_immutable": prior_audit,
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "context": {
            "context_id": CONTEXT_ID,
            "family": CONTEXT_FAMILY,
            "file": context_ref["file"],
            "file_sha256": context_ref["sha256"],
            "parameters_sha256": context["parameters_sha256"],
            "selected_candidate_index": selected["candidate_index"],
            "selected_swing_underresponse_gain": selected["swing_underresponse_gain"],
            "calibration_evaluation_seeds": selected["evaluation_seeds"],
            "base_off_success_rate": selected["off_success_rate"],
            "base_on_success_rate": selected["on_success_rate"],
            "alignment_coverage": selected["alignment_coverage"],
            "shield_rescue_rate": selected["shield_rescue_rate"],
            "base_policy_only_first_qualifier": True,
            "adapted_outcomes_used_for_selection": False,
        },
        "calibration_evidence": {**calibration_ref, "checks": calibration_checks},
        "training": formal_algorithm_parameters(),
        "learning_semantics": {
            "single_actor": True,
            "single_privileged_critic": True,
            "runtime_cbf_executes_filtered_action": True,
            "ppo_stores_raw_sampled_action_and_behavior_log_probability": True,
            "teacher_actor_coordinates_prevent_double_plant_scaling": True,
            "teacher_target_stop_gradient": True,
            "teacher_success_gate_fixed_before_training": True,
            "moving_reference_round_start_pi_k": True,
            "one_on_policy_batch_per_round": True,
        },
        "rollback": {
            "allowed_reasons": [
                "non-finite actor/critic/loss/gradient state",
                "moving forward KL above 0.01",
                "raw-action or behavior-Gaussian routing corruption",
                "teacher telemetry/reprojection/swing-selection corruption",
                "actor or critic optimizer-state corruption",
            ],
            "performance_rollback_forbidden": True,
        },
        "excluded": {
            "additional_actor_observations": True,
            "specialist_reward": True,
            "failure_or_success_bank": True,
            "state_restart": True,
            "candidate_line_search": True,
            "performance_gate_or_best_checkpoint": True,
            "multiple_critics_or_risk_head": True,
            "multiple_contexts_or_adaptation_seeds": True,
        },
        "final_policy": {
            "rule": "round 8 actor, independent of performance",
            "round_checkpoints_recovery_and_curve_only": True,
        },
        "evaluation": {
            "conditions": ["pi0_off", "pi0_on", "pi8_on", "pi8_off"],
            "episodes_per_condition": FINAL_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "seed_base": FINAL_SEED_BASE,
            "same_initial_conditions_all_four_conditions": True,
            "deterministic_policy_mean": True,
        },
        "development_gate": {
            "minimum_off_success_delta": MINIMUM_OFF_SUCCESS_DELTA,
            "minimum_on_success_delta": MINIMUM_ON_SUCCESS_DELTA,
            "off_kick_rate_must_strictly_decrease": True,
            "minimum_on_intervention_per_riser_relative_reduction": (
                MINIMUM_INTERVENTION_REDUCTION
            ),
            "point_estimates_only": True,
            "used_for_training_rollback_stopping_or_selection": False,
        },
        "randomness_preflight": randomness,
        "fresh_execution_seeds": {
            "adaptation_seed": ADAPTATION_SEED,
            "final_seed_base": FINAL_SEED_BASE,
        },
        "prospective_execution": {
            "calibration_completed": True,
            "adaptation_started": False,
            "final_evaluation_started": False,
            "adapted_policy_outcomes_observed": False,
            "fresh_adaptation_count_planned": 1,
            "outcome_driven_rerun_forbidden": True,
        },
    }
    _write_immutable(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
