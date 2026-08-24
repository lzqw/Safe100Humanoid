"""Prospectively freeze the single fixed v29 experiment before its smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v29_protocol import (
    ADAPTATION_SEED,
    BASE_CHECKPOINT_SHA256,
    CONDITIONS,
    D0_CONDITIONS,
    D0_EPISODES,
    D0_SEED_BASE,
    EXPERIMENT_NAME,
    FINAL_EPISODES,
    FINAL_SEED_BASE,
    POLICY_METHOD,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
    SOURCE_FILES,
    fixed_environment_parameters,
    formal_algorithm_parameters,
)
from proximal_v23_io import file_sha256


HISTORICAL_VERSIONS = ("v25", "v26", "v27", "v28")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_source_hashes(repo: Path, commit: str) -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if content != committed:
            raise RuntimeError(f"v29 source is not committed: {relative}")
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def _historical_result_boundary(repo: Path, commit: str) -> dict[str, Any]:
    trees = {}
    for version in HISTORICAL_VERSIONS:
        relative = f"results/online/proximal_{version}"
        if not (repo / relative).is_dir():
            raise FileNotFoundError(repo / relative)
        trees[version] = _git(repo, "rev-parse", f"{commit}:{relative}")
    return {
        "versions": list(HISTORICAL_VERSIONS),
        "git_trees": trees,
        "unchanged_at_freeze": True,
        "rerun_recomputed_or_reinterpreted": False,
    }


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
    if output.exists():
        raise RuntimeError(f"refusing to overwrite an existing v29 freeze: {output}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v29 freeze requires a clean committed worktree")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v29 base checkpoint differs from fixed pi0")
    commit = _git(repo, "rev-parse", "HEAD")
    sources = _committed_source_hashes(repo, commit)
    historical = _historical_result_boundary(repo, commit)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": "fixed_before_v29_smoke_and_adaptation",
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": sources,
            "all_sources_committed_before_smoke": True,
        },
        "prior_results_immutable": historical,
        "base_checkpoint_sha256": file_sha256(checkpoint),
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "environment": fixed_environment_parameters(),
        "training": formal_algorithm_parameters(),
        "learning_semantics": {
            "single_actor_and_single_privileged_critic": True,
            "original_actor_and_critic_observation_interfaces": True,
            "runtime_cbf_always_on_during_training": True,
            "ppo_stores_raw_action_and_raw_behavior_log_probability": True,
            "cbf_safe_action_is_executed_only": True,
            "teacher_target_actor_coordinate_plant": "identity",
            "teacher_target_stop_gradient": True,
            "teacher_labels_vectorized_on_gpu": True,
            "teacher_label_python_loops_at_most_horizon_offsets": True,
            "teacher_loss_normalization": "sum_weights_plus_1e-8",
            "round_reference": "deepcopy_round_start_actor",
        },
        "smoke": {
            "seed": SMOKE_SEED,
            "num_envs": SMOKE_ENVS,
            "steps": SMOKE_STEPS,
            "updates": 1,
            "must_have_nonzero_eligible_teacher_count": True,
        },
        "evaluation": {
            "target_conditions_in_order": [item[0] for item in CONDITIONS],
            "target_paired_identities": FINAL_EPISODES,
            "target_seed_base": FINAL_SEED_BASE,
            "D0_conditions_in_order": [item[0] for item in D0_CONDITIONS],
            "D0_paired_identities": D0_EPISODES,
            "D0_seed_base": D0_SEED_BASE,
            "preferred_batch_size": PREFERRED_EVAL_BATCH_SIZE,
            "memory_only_fallback_batch_size": 128,
            "deterministic_policy_mean": True,
            "paired_95_ci_is_not_a_gate": True,
        },
        "effect_interpretation": {
            "clearly_effective": {
                "minimum_target_on_success_delta": 0.03,
                "minimum_target_off_success_delta": 0.05,
                "off_kick_rate_decreases": True,
                "on_intervention_per_riser_decreases": True,
                "minimum_D0_on_success_delta": -0.05,
            },
            "partially_effective": {
                "target_on_success_non_decreasing": True,
                "minimum_internalization_metrics_improved": 2,
            },
            "conditional_followup": "teacher-weight-zero only after clearly effective",
        },
        "rollback": {
            "allowed_reasons": [
                "non-finite value",
                "moving forward KL above 0.01",
                "raw/safe action or behavior-logprob routing error",
                "teacher transform error above 1e-6",
            ],
            "performance_rollback_forbidden": True,
        },
        "final_policy": {
            "rule": "unconditional round 8 actor",
            "candidate_best_or_checkpoint_selection": False,
        },
        "excluded": {
            "extra_reward_observation_noise_gain_or_drift": True,
            "failure_or_success_replay": True,
            "multiple_seeds_or_contexts": True,
            "pre_main_no_teacher_control": True,
            "parameter_or_checkpoint_sweep": True,
            "full_step_telemetry_in_git": True,
        },
        "fresh_execution_seeds": {
            "smoke": SMOKE_SEED,
            "adaptation": ADAPTATION_SEED,
            "final_base": FINAL_SEED_BASE,
            "D0_base": D0_SEED_BASE,
        },
        "prospective_execution": {
            "smoke_started": False,
            "adaptation_started": False,
            "final_evaluation_started": False,
            "outcomes_observed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "config_sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
