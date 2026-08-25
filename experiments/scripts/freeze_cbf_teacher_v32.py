"""Freeze the v32 protocol before any v32 simulator execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v32_protocol import (
    BASE_CHECKPOINT_SHA256,
    BOOTSTRAP_SEED_BASE,
    CONTINUATION_FINAL_ROUND,
    CONTINUATION_SCHEDULES,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    MIXED_CONTEXT_CAPACITY,
    MIXED_D0_SEED_BASE,
    MIXED_FINAL_ROUND,
    MIXED_SCHEDULE,
    MONITOR_EPISODES,
    MONITOR_ROUNDS,
    MONITOR_SEED_BASES,
    POLICY_METHOD,
    PREFLIGHT_CASES,
    PROTOCOL_ID,
    SOURCE_FILES,
    V31_A2_ROUND8_SHA256,
    a2_parameters,
    common_algorithm_parameters,
    environment_parameters,
    learning_rates,
    mixed_context_env_counts,
    run_matrix,
)
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _schedule_payload(kind: str, schedule: str, rounds: range) -> list[dict[str, Any]]:
    return [
        {
            "round": round_index,
            "actor_learning_rate": learning_rates(kind, schedule, round_index)[0],
            "critic_learning_rate": learning_rates(kind, schedule, round_index)[1],
        }
        for round_index in rounds
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = args.base_checkpoint.resolve()
    v31_root = args.v31_formal_root.resolve()
    output = args.output.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v32 freeze requires a clean committed worktree")
    if output.exists():
        raise RuntimeError(f"v32 protocol output already exists: {output}")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v32 base checkpoint hash differs")

    v31_manifest_path = (
        repo / "results/online/proximal_v31/external_artifacts_manifest.json"
    )
    v31_result_path = repo / "results/online/proximal_v31/formal/combined_results.json"
    v31_manifest = json.loads(v31_manifest_path.read_text())
    v31_result = json.loads(v31_result_path.read_text())
    manifest_hashes = {
        item["path"]: item["sha256"]
        for item in v31_manifest["logical_roots"]["training"]["files"]
    }
    continuation_inputs = {}
    for context in FORMAL_CONTEXTS:
        logical = f"{context}/A2/round_08.pt"
        checkpoint = v31_root / context / "A2" / "round_08.pt"
        expected = V31_A2_ROUND8_SHA256[context]
        checks = {
            "exists": checkpoint.is_file(),
            "external_hash": checkpoint.is_file()
            and file_sha256(checkpoint) == expected,
            "manifest_hash": manifest_hashes.get(logical) == expected,
            "v31_result_actor": v31_result["contexts"][context]["target"]["A2_on"][
                "checkpoint_sha256"
            ]
            == expected,
        }
        if not all(checks.values()):
            raise RuntimeError(f"invalid v31 continuation input {context}: {checks}")
        continuation_inputs[context] = {
            "logical_external_path": logical,
            "sha256": expected,
            "bytes": checkpoint.stat().st_size,
            "v31_policy": "A2 unconditional round 8",
        }

    commit = _git(repo, "rev-parse", "HEAD")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = file_sha256(path)
    prior_versions = [f"v{number}" for number in range(25, 32)]
    prior_trees = {
        version: _git(repo, "rev-parse", f"{commit}:results/online/proximal_{version}")
        for version in prior_versions
    }

    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_v32_preflight_and_formal",
        "policy_method": POLICY_METHOD,
        "objective": (
            "Measure whether three-times online interaction improves v31 A2 and "
            "whether learning-rate decay reduces late-round drift."
        ),
        "base_checkpoint": {
            "logical_role": "common pi0 and mixed-policy initialization",
            "sha256": BASE_CHECKPOINT_SHA256,
            "bytes": base.stat().st_size,
        },
        "v31_reference": {
            "result_sha256": file_sha256(v31_result_path),
            "external_manifest_sha256": file_sha256(v31_manifest_path),
            "highest_mean_CBF_on_method": v31_result["three_context_summary"][
                "highest_mean_CBF_on_success_method"
            ],
            "A2_round8_inputs": continuation_inputs,
        },
        "contexts": {
            context: environment_parameters(context)
            for context in (*FORMAL_CONTEXTS, "D0")
        },
        "algorithm": common_algorithm_parameters(),
        "A2_configuration_unchanged": a2_parameters(),
        "formal_runs": run_matrix(),
        "learning_rate_schedules": {
            "continuation": {
                schedule: _schedule_payload(
                    "continuation", schedule, range(9, CONTINUATION_FINAL_ROUND + 1)
                )
                for schedule in CONTINUATION_SCHEDULES
            },
            "mixed": _schedule_payload(
                "mixed", MIXED_SCHEDULE, range(1, MIXED_FINAL_ROUND + 1)
            ),
        },
        "mixed_context": {
            "underlying_envs_per_context": MIXED_CONTEXT_CAPACITY,
            "exposed_envs_per_round": 64,
            "episode_context_fixed": True,
            "round_assignments": {
                str(round_index): mixed_context_env_counts(round_index)
                for round_index in range(1, MIXED_FINAL_ROUND + 1)
            },
            "three_round_env_totals": {context: 64 for context in FORMAL_CONTEXTS},
            "twenty_four_round_env_totals": {
                context: sum(
                    mixed_context_env_counts(round_index)[context]
                    for round_index in range(1, MIXED_FINAL_ROUND + 1)
                )
                for context in FORMAL_CONTEXTS
            },
        },
        "preflight": {
            "cases": list(PREFLIGHT_CASES),
            "functional_only_not_performance_evidence": True,
            "simulator_case_rerun_after_pass": False,
        },
        "monitor": {
            "rounds": list(MONITOR_ROUNDS),
            "target_episodes": MONITOR_EPISODES,
            "runtime_CBF": "on",
            "deterministic_policy_mean": True,
            "seed_bases": MONITOR_SEED_BASES,
            "used_for_selection_stopping_or_training": False,
        },
        "formal_evaluation": {
            "target_episodes_per_condition": FORMAL_TARGET_EPISODES,
            "D0_episodes_per_condition": FORMAL_D0_EPISODES,
            "target_modes": ["on", "off"],
            "continuation_D0_mode": "on",
            "target_seed_bases": FORMAL_TARGET_SEED_BASES,
            "continuation_D0_seed_bases": FORMAL_D0_SEED_BASES,
            "mixed_D0_seed_base": MIXED_D0_SEED_BASE,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
            "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
            "no_pass_fail_gate": True,
            "no_checkpoint_selection": True,
        },
        "stop_reasons": {
            "allowed": [
                "NaN_or_Inf",
                "optimizer_state_corruption",
                "raw_action_storage_error",
                "safe_action_execution_routing_error",
                "simulator_crash",
                "infrastructure_interruption",
            ],
            "performance_or_KL_stop_forbidden": True,
        },
        "prior_results_immutable": {
            "versions": prior_versions,
            "git_trees": prior_trees,
            "v31_result_recomputed_or_modified": False,
        },
        "source_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
        },
    }
    if protocol["v31_reference"]["highest_mean_CBF_on_method"] != "A2":
        raise RuntimeError("v32 premise differs from published v31 result")
    if any(
        value != 512
        for value in protocol["mixed_context"]["twenty_four_round_env_totals"].values()
    ):
        raise RuntimeError(
            "v32 mixed 24-round context allocation is not exactly balanced"
        )
    _atomic_json(output, protocol)
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_id": PROTOCOL_ID,
                "source_commit": commit,
                "formal_runs": len(protocol["formal_runs"]),
                "source_files": len(source_hashes),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
