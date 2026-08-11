"""Freeze v24 implementation and base-only calibration before any episode."""

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

from proximal_v23_io import file_sha256
from proximal_v24_protocol import (
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_CANDIDATE_PARAMETER_SEEDS,
    CALIBRATION_EPISODES,
    CALIBRATION_FRICTIONS,
    CALIBRATION_MINIMUM_FALLS,
    CALIBRATION_MINIMUM_PURITY,
    CALIBRATION_REPEATS,
    CALIBRATION_SUCCESS_BOUNDS,
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
    V23_FINAL_TEST_SHA256,
    V23_FROZEN_RESULT,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    calibration_evaluation_seed,
    canonical_sha256,
    formal_algorithm_parameters,
    fresh_randomness_report,
    pure_contact_context_audit,
)

from src.tasks.stairs_cbf.deployment_context import (
    generate_v22_specialist_context,
    validate_frozen_deployment_context,
)

SOURCE_FILES = (
    "docs/CBF_PROXIMAL_V24_CONTACT_COMPLETION.md",
    "src/tasks/stairs_cbf/actions.py",
    "src/tasks/stairs_cbf/command.py",
    "src/tasks/stairs_cbf/config.py",
    "src/tasks/stairs_cbf/deployment_context.py",
    "src/tasks/stairs_cbf/hard_cases.py",
    "src/tasks/stairs_cbf/mdp.py",
    "src/tasks/stairs_cbf/proximal.py",
    "src/tasks/stairs_cbf/proximal_context.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/proximal_v23_protocol.py",
    "experiments/scripts/refine_proximal_v23.py",
    "experiments/scripts/proximal_v24_protocol.py",
    "experiments/scripts/evaluate_proximal_v24.py",
    "experiments/scripts/calibrate_proximal_v24.py",
    "experiments/scripts/refine_proximal_v24.py",
    "experiments/scripts/audit_proximal_v24.py",
    "experiments/scripts/plot_proximal_v24.py",
    "experiments/scripts/verify_proximal_v24.py",
    "experiments/scripts/build_proximal_completion.py",
    "experiments/scripts/freeze_proximal_v24_precalibration.py",
    "experiments/scripts/freeze_proximal_v24_protocol.py",
    "experiments/scripts/run_proximal_v24.sh",
    "experiments/tests/test_proximal_v24.py",
)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_committed_sources(repo: Path, commit: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_bytes()
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if blob != content:
            raise RuntimeError(
                f"v24 prospective source is not committed at {commit}: {relative}"
            )
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def _v23_immutable_audit(repo: Path, commit: str) -> dict[str, Any]:
    protocol = repo / "results/online/proximal_v23/protocol.json"
    final = repo / "results/online/proximal_v23/final/final_test.json"
    payload = json.loads(final.read_text())
    actual_values = {
        "target_base_success": payload["target"]["success"]["baseline_mean"],
        "target_final_success": payload["target"]["success"]["final_mean"],
        "target_success_delta": payload["target"]["success"]["delta"],
        "target_fall_delta": payload["target"]["fall"]["delta"],
        "target_repairs": payload["target"]["repairs_regressions"]["repair_count"],
        "target_regressions": payload["target"]["repairs_regressions"][
            "regression_count"
        ],
    }
    result_tree = _git_output(
        repo, "rev-parse", f"{commit}:results/online/proximal_v23"
    )
    checks = {
        "protocol_sha256": file_sha256(protocol) == V23_PROTOCOL_SHA256,
        "final_test_sha256": file_sha256(final) == V23_FINAL_TEST_SHA256,
        "entire_result_git_tree": result_tree == V23_RESULT_GIT_TREE,
        "frozen_values": actual_values == V23_FROZEN_RESULT,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v23 immutable boundary changed: {checks}")
    return {
        "unchanged": True,
        "checks": checks,
        "protocol_sha256": file_sha256(protocol),
        "final_test_sha256": file_sha256(final),
        "result_git_tree": result_tree,
        "negative_result": actual_values,
        "rerun_or_recomputed": False,
    }


def _candidate_grid() -> list[dict[str, Any]]:
    output = []
    for index, (seed, friction) in enumerate(
        zip(
            CALIBRATION_CANDIDATE_PARAMETER_SEEDS,
            CALIBRATION_FRICTIONS,
            strict=True,
        )
    ):
        context = validate_frozen_deployment_context(
            generate_v22_specialist_context(CONTEXT_ID, seed)
        )
        audit = pure_contact_context_audit(context, candidate_index=index)
        output.append(
            {
                "candidate_index": index,
                "candidate_parameter_seed": seed,
                "foot_friction": friction,
                "parameters_sha256": context["parameters_sha256"],
                "canonical_context_sha256": canonical_sha256(context),
                "pure_context_audit": audit,
                "evaluation_seeds": [
                    calibration_evaluation_seed(index, repeat)
                    for repeat in range(CALIBRATION_REPEATS)
                ],
            }
        )
    return output


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freezing v24")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v24 base checkpoint differs from frozen pi0")
    if (repo / "results/online/proximal_v24").exists():
        raise RuntimeError("v24 result path exists before the pre-calibration freeze")
    source_hashes = _verify_committed_sources(repo, commit)
    v23_audit = _v23_immutable_audit(repo, commit)
    candidates = _candidate_grid()
    randomness = fresh_randomness_report(repo)
    if not randomness["passed"]:
        raise RuntimeError(
            f"v24 fresh randomness collision: {randomness['collisions']}"
        )

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": "prospectively_frozen_before_v24_base_only_calibration",
        "experiment_class": (
            "independent single pure-low-friction contact completion test"
        ),
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
            "all_execution_sources_committed_before_first_v24_episode": True,
        },
        "v23_lateral_immutable": v23_audit,
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "context_family": {
            "context_id": CONTEXT_ID,
            "mode": CONTEXT_MODE,
            "family": CONTEXT_FAMILY,
            "only_changed_physical_axis": "foot_friction",
            "ordered_light_to_severe": True,
            "candidate_grid": candidates,
        },
        "calibration": {
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "first_qualifying_candidate_is_frozen": True,
            "candidate_parameter_seeds": list(CALIBRATION_CANDIDATE_PARAMETER_SEEDS),
            "candidate_foot_frictions": list(CALIBRATION_FRICTIONS),
            "episodes_per_candidate": CALIBRATION_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "repeats_per_candidate": CALIBRATION_REPEATS,
            "success_rate_bounds_inclusive": list(CALIBRATION_SUCCESS_BOUNDS),
            "minimum_fall_count": CALIBRATION_MINIMUM_FALLS,
            "minimum_contact_purity": CALIBRATION_MINIMUM_PURITY,
            "purity_primary_denominator": "all non-success episodes",
            "falls_only_purity_also_required": True,
            "ordinary_cbf_intervention_is_failure": False,
            "outcome_dependent_reselection_forbidden": True,
        },
        "training": formal_algorithm_parameters(),
        "learning_semantics": {
            "actor_observation_dim": 405,
            "critic_observation_dim": 838,
            "single_actor": True,
            "single_privileged_critic": True,
            "runtime_cbf_executes_filtered_action": True,
            "ppo_stores_raw_policy_action": True,
            "moving_reference": "current round-start pi_k",
            "final_policy": "unconditional round-8 actor",
            "performance_rollback_or_selection": False,
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
        },
        "development_gate": {
            "minimum_target_success_delta": MINIMUM_TARGET_SUCCESS_DELTA,
            "maximum_target_fall_delta": MAXIMUM_TARGET_FALL_DELTA,
            "minimum_D0_success_delta": MINIMUM_D0_SUCCESS_DELTA,
            "point_estimates_only": True,
            "used_for_training_or_selection": False,
        },
        "excluded": {
            "additional_actor_observations": True,
            "specialist_reward": True,
            "failure_or_success_bank": True,
            "state_restart": True,
            "candidate_line_search": True,
            "performance_gate_or_best_checkpoint": True,
            "multiple_critics_or_risk_head": True,
            "multi_context_or_multiple_adaptation_seed": True,
            "off_diagonal_macro_or_candidate_ablation": True,
        },
        "randomness_preflight": randomness,
        "fresh_execution_seeds": {
            "adaptation_seed": ADAPTATION_SEED,
            "final_target_seed_start": FINAL_TARGET_SEED,
            "final_D0_seed_start": FINAL_D0_SEED,
            "report_bootstrap_seeds": REPORT_BOOTSTRAP_SEEDS,
        },
        "prospective_execution": {
            "calibration_started": False,
            "adaptation_started": False,
            "final_evaluation_started": False,
            "adapted_policy_outcomes_observed": False,
            "fresh_adaptation_count_planned": 1,
        },
    }
    _write_immutable(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
