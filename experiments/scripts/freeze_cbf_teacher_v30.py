"""Prospectively freeze v30 before development and again before formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v30_protocol import (
    ARMS,
    BASE_CHECKPOINT_SHA256,
    BOOTSTRAP_SEED_BASE,
    DEVELOPMENT_ADAPTATION_SEED,
    DEVELOPMENT_D0_EPISODES,
    DEVELOPMENT_D0_SEED_BASE,
    DEVELOPMENT_TARGET_EPISODES,
    DEVELOPMENT_TARGET_SEED_BASE,
    FALLBACK_EVAL_BATCH_SIZE,
    FORMAL_ADAPTATION_SEEDS,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    MONITOR_EPISODES,
    MONITOR_SEED,
    POLICY_METHOD,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    SMOKE_ENVS,
    SMOKE_SEED,
    SMOKE_STEPS,
    SOURCE_FILES,
    TEACHER_ARMS,
    arm_parameters,
    common_training_parameters,
    environment_parameters,
)
from proximal_v23_io import file_sha256

HISTORICAL_VERSIONS = ("v25", "v26", "v27", "v28", "v29")


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
        content = path.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if content != committed:
            raise RuntimeError(f"v30 source is not committed: {relative}")
        output[relative] = hashlib.sha256(content).hexdigest()
    return output


def _historical_boundary(repo: Path, commit: str) -> dict[str, Any]:
    trees = {
        version: _git(
            repo,
            "rev-parse",
            f"{commit}:results/online/proximal_{version}",
        )
        for version in HISTORICAL_VERSIONS
    }
    return {
        "versions": list(HISTORICAL_VERSIONS),
        "git_trees": trees,
        "unchanged_at_freeze": True,
        "rerun_recomputed_or_reinterpreted": False,
    }


def _common_payload(repo: Path, checkpoint: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
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
            "main_riser_height_m": 0.18,
            "clearance_barrier_slope": 0.8,
            "post_edge_recovery_window_m": 0.15,
            "exponential_cbf_alpha": 10.0,
            "parameter_or_checkpoint_sweep": False,
        },
        "common_training": common_training_parameters(),
        "development_arms": {arm: arm_parameters(arm) for arm in ARMS},
        "contexts": {
            name: environment_parameters(name)
            for name in ("DEV", "F1", "F2", "F3", "D0")
        },
        "learning_semantics": {
            "single_actor_405D": True,
            "single_privileged_critic_838D": True,
            "runtime_cbf_on_during_training": True,
            "execute_safe_store_raw_action_and_log_probability": True,
            "residual_target": "round_reference_mean + eta * (safe_sample - raw_sample)",
            "teacher_target_stop_gradient": True,
            "weight": "intervention * clip(correction_norm / 0.05, 0, 1)",
            "smooth_l1_beta": 0.05,
            "per_action_dimension_mean": True,
            "weight_normalization": "sum_weights_plus_1e-8",
        },
        "smoke": {
            "attempt_limit": 1,
            "seed": SMOKE_SEED,
            "num_envs": SMOKE_ENVS,
            "steps": SMOKE_STEPS,
            "actor_epochs": 1,
            "critic_epochs": 1,
            "kl_threshold_gate": False,
            "task_performance_gate": False,
        },
        "development": {
            "context": "DEV",
            "adaptation_seed": DEVELOPMENT_ADAPTATION_SEED,
            "arms_in_order": list(ARMS),
            "target_episodes": DEVELOPMENT_TARGET_EPISODES,
            "target_seed_base": DEVELOPMENT_TARGET_SEED_BASE,
            "D0_episodes": DEVELOPMENT_D0_EPISODES,
            "D0_seed_base": DEVELOPMENT_D0_SEED_BASE,
            "base_evaluated_once_and_cached": True,
            "selection_candidates": list(TEACHER_ARMS),
            "selection_rule": [
                "maximum round-8 target CBF-on success",
                "maximum round-8 target CBF-off success",
                "minimum round-8 target interventions per riser",
            ],
            "A0_excluded_from_selection": True,
            "KL_and_intermediate_checkpoints_excluded_from_selection": True,
        },
        "evaluation_batching": {
            "preferred": PREFERRED_EVAL_BATCH_SIZE,
            "memory_only_fallback": FALLBACK_EVAL_BATCH_SIZE,
            "fallback_changes_only_batching_not_identity": True,
        },
        "stopping": {
            "allowed_reasons": [
                "NaN or Inf",
                "raw/safe action or behavior-logprob routing error",
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
    }


def _freeze_development(
    repo: Path, checkpoint: Path, output: Path, external_root: Path | None
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite v30 development freeze: {output}")
    if external_root is not None and external_root.exists():
        markers = list(external_root.rglob("execution_started.json"))
        if markers:
            raise RuntimeError(
                "v30 execution already started before development freeze"
            )
    payload = _common_payload(repo, checkpoint)
    payload.update(
        {
            "status": "frozen_before_v30_development",
            "formal": {
                "selected_teacher": None,
                "contexts": list(FORMAL_CONTEXTS),
                "adaptation_seeds": dict(FORMAL_ADAPTATION_SEEDS),
                "target_seed_bases": dict(FORMAL_TARGET_SEED_BASES),
                "D0_seed_bases": dict(FORMAL_D0_SEED_BASES),
                "selection_not_yet_observed": True,
            },
            "prospective_execution": {
                "smoke_started": False,
                "development_started": False,
                "development_outcomes_observed": False,
                "formal_started": False,
            },
        }
    )
    return payload


def _freeze_formal(
    repo: Path,
    checkpoint: Path,
    output: Path,
    development_audit: Path,
    external_root: Path | None,
) -> dict[str, Any]:
    if not output.is_file() or not development_audit.is_file():
        raise FileNotFoundError(
            "formal freeze requires prior protocol and development audit"
        )
    previous = json.loads(output.read_text())
    audit = json.loads(development_audit.read_text())
    if previous.get("status") != "frozen_before_v30_development":
        raise RuntimeError("v30 prior protocol is not the development freeze")
    if audit.get("protocol_id") != PROTOCOL_ID or not audit.get("complete"):
        raise RuntimeError("v30 development audit is incomplete")
    selected = audit.get("selected_teacher", {})
    arm = selected.get("arm")
    if arm not in TEACHER_ARMS or selected.get("configuration") != arm_parameters(arm):
        raise RuntimeError("v30 selected teacher is not a frozen teacher arm")
    current_sources = _source_hashes(repo, _git(repo, "rev-parse", "HEAD"))
    if current_sources != previous["source_boundary"]["source_files"]:
        raise RuntimeError("v30 source changed after development freeze")
    current_trees = _historical_boundary(repo, _git(repo, "rev-parse", "HEAD"))[
        "git_trees"
    ]
    if current_trees != previous["prior_results_immutable"]["git_trees"]:
        raise RuntimeError("v25-v29 result tree changed before formal freeze")
    if external_root is not None:
        for context in FORMAL_CONTEXTS:
            for arm_name in ("A0", arm):
                if (external_root / "formal" / context / arm_name).exists():
                    raise RuntimeError("v30 formal execution already started")
    payload = previous
    payload["status"] = "frozen_before_v30_formal"
    payload["formal"] = {
        "selected_teacher": {
            "arm": arm,
            "configuration": arm_parameters(arm),
            "selection_record_sha256": file_sha256(development_audit),
            "development_source": str(development_audit),
        },
        "contexts": list(FORMAL_CONTEXTS),
        "adaptation_seeds": dict(FORMAL_ADAPTATION_SEEDS),
        "target_episodes_per_context": FORMAL_TARGET_EPISODES,
        "target_seed_bases": dict(FORMAL_TARGET_SEED_BASES),
        "D0_episodes_per_context": FORMAL_D0_EPISODES,
        "D0_seed_bases": dict(FORMAL_D0_SEED_BASES),
        "conditions": [
            "base_off",
            "base_on",
            "control_on",
            "control_off",
            "teacher_on",
            "teacher_off",
        ],
        "D0_conditions": ["base_on", "control_on", "teacher_on"],
        "bootstrap_samples_max": FORMAL_BOOTSTRAP_SAMPLES,
        "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
        "paired_CI_not_a_gate": True,
        "final_audit_cannot_select_or_modify_policy": True,
        "checkpoint_monitor": {
            "context": "F1",
            "episodes": MONITOR_EPISODES,
            "seed": MONITOR_SEED,
            "rounds": list(range(9)),
            "methods": ["control", "teacher"],
            "filter_modes": ["on", "off"],
            "read_only_after_all_training": True,
            "selection_gate": False,
        },
    }
    payload["prospective_execution"].update(
        {
            "smoke_started": True,
            "development_started": True,
            "development_outcomes_observed": True,
            "formal_started": False,
            "formal_outcomes_observed": False,
        }
    )
    payload["development_binding"] = {
        "audit": str(development_audit),
        "sha256": file_sha256(development_audit),
        "selected_arm": arm,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "formal"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path)
    parser.add_argument("--external-root", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output.resolve()
    external_root = None if args.external_root is None else args.external_root.resolve()
    if not checkpoint.is_file() or file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v30 base checkpoint differs from fixed pi0")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v30 freeze requires a clean committed worktree")
    if args.stage == "development":
        payload = _freeze_development(repo, checkpoint, output, external_root)
    else:
        if args.development_audit is None:
            raise ValueError("formal freeze requires --development-audit")
        payload = _freeze_formal(
            repo,
            checkpoint,
            output,
            args.development_audit.resolve(),
            external_root,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
