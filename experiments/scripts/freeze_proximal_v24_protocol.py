"""Freeze selected v24 contact context before the sole formal adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freeze_proximal_v24_precalibration import (
    _v23_immutable_audit,
    _verify_committed_sources,
)
from proximal_v23_io import file_sha256
from proximal_v24_protocol import (
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    CONTEXT_MODE,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    FINAL_D0_EPISODES,
    FINAL_D0_SEED,
    FINAL_TARGET_EPISODES,
    FINAL_TARGET_SEED,
    MAXIMUM_TARGET_FALL_DELTA,
    MINIMUM_D0_SUCCESS_DELTA,
    MINIMUM_TARGET_SUCCESS_DELTA,
    POLICY_METHOD,
    PROTOCOL_ID,
    REPORT_BOOTSTRAP_SAMPLES,
    REPORT_BOOTSTRAP_SEEDS,
    formal_algorithm_parameters,
    fresh_randomness_report,
    pure_contact_context_audit,
    validate_v24_calibrated_context,
)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_file_audit(repo: Path, path: Path, commit: str) -> dict[str, str]:
    relative = path.relative_to(repo)
    content = path.read_bytes()
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if content != blob:
        raise RuntimeError(f"formal v24 input is not committed: {relative}")
    return {
        "file": str(relative),
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_commit": commit,
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise RuntimeError(f"refusing to overwrite another v24 protocol: {path}")
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
    precalibration_path = args.precalibration_protocol.resolve()
    context_path = args.context.resolve()
    calibration_path = args.calibration_summary.resolve()
    formal_output_dir = args.formal_output_dir.resolve()
    output = args.output.resolve()
    for path in (checkpoint, precalibration_path, context_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before formal v24 freeze")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("formal v24 base checkpoint differs from frozen pi0")
    if (formal_output_dir / "formal_execution_started.json").exists():
        raise RuntimeError("v24 adaptation was started before the formal freeze")
    if (repo / "results/online/proximal_v24/final/final_test.json").exists():
        raise RuntimeError("v24 final outcome exists before the formal freeze")

    source_hashes = _verify_committed_sources(repo, commit)
    precalibration_reference = _committed_file_audit(repo, precalibration_path, commit)
    context_reference = _committed_file_audit(repo, context_path, commit)
    calibration_reference = _committed_file_audit(repo, calibration_path, commit)
    precalibration = json.loads(precalibration_path.read_text())
    context = validate_v24_calibrated_context(json.loads(context_path.read_text()))
    calibration = json.loads(calibration_path.read_text())
    selected = context["calibration"]["attempts"][-1]
    calibration_checks = {
        "precalibration_id": precalibration.get("protocol_id") == PROTOCOL_ID,
        "precalibration_status": precalibration.get("status")
        == "prospectively_frozen_before_v24_base_only_calibration",
        "base_only": calibration.get("base_policy_only") is True,
        "adapted_policy_absent": calibration.get("adapted_policy_evaluations_used")
        is False,
        "first_qualifier": calibration.get("candidate_count_evaluated")
        == len(context["calibration"]["attempts"]),
        "context_file": calibration.get("frozen_context_file_sha256")
        == file_sha256(context_path),
        "context_parameters": calibration.get("parameters_sha256")
        == context["parameters_sha256"],
        "selected_seed": calibration.get("selected_candidate_parameter_seed")
        == selected["candidate_parameter_seed"],
        "selected_friction": calibration.get("selected_foot_friction")
        == selected["candidate_foot_friction"],
        "qualifies": selected.get("qualifies") is True,
        "precalibration_sha": context["calibration"].get(
            "precalibration_protocol_sha256"
        )
        == file_sha256(precalibration_path),
    }
    if not all(calibration_checks.values()):
        raise RuntimeError(f"v24 formal calibration mismatch: {calibration_checks}")
    pure_audit = pure_contact_context_audit(
        context,
        candidate_index=int(selected["candidate_index"]),
    )
    randomness = fresh_randomness_report(repo)
    if not randomness["passed"]:
        raise RuntimeError(
            f"v24 fresh randomness collision: {randomness['collisions']}"
        )
    v23_audit = _v23_immutable_audit(repo, commit)

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": (
            "prospectively_frozen_after_base_only_calibration_before_adaptation"
        ),
        "experiment_class": (
            "independent single pure-low-friction contact completion test"
        ),
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
            "precalibration_protocol": precalibration_reference,
            "calibrated_context": context_reference,
            "calibration_summary": calibration_reference,
            "all_formal_sources_and_selected_context_committed_before_adaptation": True,
        },
        "v23_lateral_immutable": v23_audit,
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "context": {
            "context_id": CONTEXT_ID,
            "mode": CONTEXT_MODE,
            "family": CONTEXT_FAMILY,
            "file": str(context_path.relative_to(repo)),
            "file_sha256": file_sha256(context_path),
            "parameters_sha256": context["parameters_sha256"],
            "selected_candidate_parameter_seed": selected["candidate_parameter_seed"],
            "selected_candidate_index": selected["candidate_index"],
            "selected_foot_friction": selected["candidate_foot_friction"],
            "calibration_evaluation_seeds": selected["evaluation_seeds"],
            "base_success_rate": selected["success_rate"],
            "fall_count": selected["fall_count"],
            "contact_purity_over_all_non_success": selected[
                "contact_purity_over_all_non_success"
            ],
            "base_policy_only_first_qualifier": True,
            "adapted_outcomes_used_for_selection": False,
            "pure_context_audit": pure_audit,
        },
        "calibration_evidence": {
            **calibration_reference,
            "checks": calibration_checks,
        },
        "training": formal_algorithm_parameters(),
        "learning_semantics": {
            "actor_observation_dim": 405,
            "critic_observation_dim": 838,
            "single_actor": True,
            "single_privileged_critic": True,
            "runtime_cbf_executes_filtered_action": True,
            "ppo_stores_raw_policy_action_and_behavior_log_probability": True,
            "moving_reference": "round-start pi_k refreshed every round",
            "reference_stop_gradient": True,
            "reward": "unchanged v23 base + fall + dual-CBF reward",
            "ordinary_cbf_intervention_is_failure": False,
            "one_on_policy_batch_per_round": True,
        },
        "rollback": {
            "allowed_reasons": [
                "non-finite actor/critic/loss/gradient state",
                "moving forward KL above 0.01",
                "raw-action or behavior-Gaussian routing corruption",
                "actor or critic optimizer-state corruption",
            ],
            "performance_rollback_forbidden": True,
        },
        "excluded": {
            "additional_actor_observations": True,
            "specialist_reward": True,
            "failure_precursor_or_matched_success_bank": True,
            "state_restart": True,
            "candidate_line_search": True,
            "performance_gate_or_best_checkpoint": True,
            "multiple_critics_or_risk_head": True,
            "multi_context_or_multiple_adaptation_seed": True,
            "off_diagonal_macro_or_candidate_ablation": True,
        },
        "final_policy": {
            "rule": "round 8 actor, independent of performance",
            "round_start_and_end_checkpoints_are_recovery_and_curve_only": True,
        },
        "evaluation": {
            "target_episodes": FINAL_TARGET_EPISODES,
            "D0_episodes": FINAL_D0_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "target_seed_start": FINAL_TARGET_SEED,
            "D0_seed_start": FINAL_D0_SEED,
            "base_and_final_conditions_paired": True,
            "deterministic_policy_mean": True,
            "runtime_cbf": True,
            "report_bootstrap_samples": REPORT_BOOTSTRAP_SAMPLES,
            "report_bootstrap_seeds": REPORT_BOOTSTRAP_SEEDS,
            "confidence_intervals_are_gates": False,
            "repairs_and_regressions_reported": True,
        },
        "development_gate": {
            "minimum_target_success_delta": MINIMUM_TARGET_SUCCESS_DELTA,
            "maximum_target_fall_delta": MAXIMUM_TARGET_FALL_DELTA,
            "minimum_D0_success_delta": MINIMUM_D0_SUCCESS_DELTA,
            "point_estimates_only": True,
            "used_for_training_rollback_stopping_or_selection": False,
        },
        "randomness_preflight": randomness,
        "fresh_execution_seeds": {
            "adaptation_seed": ADAPTATION_SEED,
            "final_target_seed_start": FINAL_TARGET_SEED,
            "D0_seed_start": FINAL_D0_SEED,
            "report_bootstrap_seeds": REPORT_BOOTSTRAP_SEEDS,
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
