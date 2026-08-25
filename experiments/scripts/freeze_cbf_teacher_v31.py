"""Prospectively freeze the complete v31 preflight and formal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    BEHAVIOR_LOG_PROB_ATOL,
    BOOTSTRAP_SEED_BASE,
    FALLBACK_EVAL_BATCH_SIZE,
    FORMAL_ADAPTATION_SEEDS,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    METHOD_ARMS,
    MONITOR_EPISODES,
    MONITOR_SEED,
    POLICY_METHOD,
    PREFERRED_EVAL_BATCH_SIZE,
    PREFLIGHT_CASES,
    PREFLIGHT_ENVS,
    PREFLIGHT_STEPS,
    PROTOCOL_ID,
    SOURCE_FILES,
    arm_parameters,
    common_training_parameters,
    environment_parameters,
)
from proximal_v23_io import file_sha256

HISTORICAL_VERSIONS = ("v25", "v26", "v27", "v28", "v29", "v30")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_hashes(repo: Path, commit: str) -> dict[str, str]:
    output = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        content = path.read_bytes()
        if content != committed:
            raise RuntimeError(f"v31 source is not committed: {relative}")
        output[relative] = hashlib.sha256(content).hexdigest()
    return output


def _historical_boundary(repo: Path, commit: str) -> dict[str, Any]:
    return {
        "versions": list(HISTORICAL_VERSIONS),
        "git_trees": {
            version: _git(
                repo, "rev-parse", f"{commit}:results/online/proximal_{version}"
            )
            for version in HISTORICAL_VERSIONS
        },
        "v30_interpretation": "immutable incomplete formal result",
        "rerun_recomputed_or_reinterpreted": False,
    }


def _payload(repo: Path, checkpoint: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    cases = [
        {"context": context, "arm": arm, "seed": seed}
        for context, arm, seed in PREFLIGHT_CASES
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_v31_preflight_and_formal",
        "policy_method": POLICY_METHOD,
        "source_boundary": {
            "git_commit": commit,
            "source_files": _source_hashes(repo, commit),
            "all_sources_committed_before_execution": True,
        },
        "prior_results_immutable": _historical_boundary(repo, commit),
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "fixed_cbf": {
            "clearance_barrier_slope": 0.8,
            "post_edge_recovery_window_m": 0.15,
            "exponential_cbf_alpha": 10.0,
            "parameter_or_checkpoint_sweep": False,
        },
        "v31_engineering_fixes": {
            "behavior_log_probability_float32_reduction_atol": BEHAVIOR_LOG_PROB_ATOL,
            "raw_action_bitwise_audit_retained": True,
            "behavior_distribution_parameter_atol_retained": 2.0e-5,
            "executed_safe_action_audit_retained": True,
            "stair_target_patch_slots": "number_of_risers + 1",
            "F3_risers": 11,
            "F3_target_patches_including_top_platform": 12,
        },
        "common_training": common_training_parameters(),
        "methods": {arm: arm_parameters(arm) for arm in METHOD_ARMS},
        "contexts": {
            name: environment_parameters(name) for name in (*FORMAL_CONTEXTS, "D0")
        },
        "preflight": {
            "attempt_limit": 1,
            "cases": cases,
            "num_envs": PREFLIGHT_ENVS,
            "steps": PREFLIGHT_STEPS,
            "actor_epochs": 1,
            "critic_epochs": 1,
            "all_contexts_constructed": True,
            "all_methods_updated": True,
            "formal_starts_immediately_after_pass": True,
        },
        "formal": {
            "contexts": list(FORMAL_CONTEXTS),
            "methods": list(METHOD_ARMS),
            "execution_order": [
                f"{context}-{arm}" for context in FORMAL_CONTEXTS for arm in METHOD_ARMS
            ],
            "adaptation_seeds": dict(FORMAL_ADAPTATION_SEEDS),
            "one_adaptation_per_context_method": True,
            "infrastructure_resume_from_latest_complete_round": True,
            "result_driven_rerun": False,
            "target_episodes_per_condition": FORMAL_TARGET_EPISODES,
            "target_seed_bases": dict(FORMAL_TARGET_SEED_BASES),
            "D0_episodes_per_condition": FORMAL_D0_EPISODES,
            "D0_seed_bases": dict(FORMAL_D0_SEED_BASES),
            "conditions": [
                "base_off",
                "base_on",
                "A0_off",
                "A0_on",
                "A1_off",
                "A1_on",
                "A2_off",
                "A2_on",
            ],
            "D0_conditions": ["base_on", "A0_on", "A1_on", "A2_on"],
            "bootstrap_samples_max": FORMAL_BOOTSTRAP_SAMPLES,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
            "paired_CI_not_a_gate": True,
            "final_audit_cannot_select_or_modify_policy": True,
            "checkpoint_monitor": {
                "context": "F1",
                "episodes": MONITOR_EPISODES,
                "seed": MONITOR_SEED,
                "rounds": list(range(9)),
                "methods": list(METHOD_ARMS),
                "filter_modes": ["on", "off"],
                "read_only_after_all_training_and_audit": True,
                "selection_gate": False,
            },
        },
        "evaluation_batching": {
            "preferred": PREFERRED_EVAL_BATCH_SIZE,
            "memory_only_fallback": FALLBACK_EVAL_BATCH_SIZE,
            "fallback_changes_only_batching_not_identity": True,
        },
        "stopping": {
            "allowed_reasons": [
                "NaN or Inf",
                "real raw/safe action or behavior-distribution routing error",
                "teacher transform error",
                "optimizer corruption",
            ],
            "target_KL_early_stop": False,
            "hard_KL_rollback": False,
            "performance_rollback": False,
        },
        "final_policy": {
            "unconditional_round_8": True,
            "candidate_best_or_performance_selection": False,
        },
        "prospective_execution": {
            "preflight_started": False,
            "formal_started": False,
            "formal_outcomes_observed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-root", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output.resolve()
    external = None if args.external_root is None else args.external_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite v31 protocol: {output}")
    if not checkpoint.is_file() or file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v31 base checkpoint differs from fixed pi0")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v31 freeze requires a clean committed worktree")
    if (
        external is not None
        and external.exists()
        and list(external.rglob("execution_started.json"))
    ):
        raise RuntimeError("v31 execution started before protocol freeze")
    payload = _payload(repo, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
