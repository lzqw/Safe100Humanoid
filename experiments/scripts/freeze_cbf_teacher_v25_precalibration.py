"""Freeze v25 implementation and ordered grid before any simulator episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v25_protocol import (
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
    FINAL_SEED_BASE,
    POLICY_METHOD,
    PROTOCOL_ID,
    SHIELD_RESCUE_MINIMUM,
    SOURCE_FILES,
    V23_FINAL_SHA256,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    V24_FINAL_SHA256,
    V24_PROTOCOL_SHA256,
    V24_RESULT_GIT_TREE,
    calibration_evaluation_seed,
    fixed_environment_parameters,
    formal_algorithm_parameters,
    fresh_randomness_report,
)
from proximal_v23_io import file_sha256


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_committed_sources(repo: Path, commit: str) -> dict[str, str]:
    hashes = {}
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
            raise RuntimeError(f"v25 source is not committed at {commit}: {relative}")
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def _prior_immutable_audit(repo: Path, commit: str) -> dict[str, Any]:
    paths = {
        "v23_protocol": repo / "results/online/proximal_v23/protocol.json",
        "v23_final": repo / "results/online/proximal_v23/final/final_test.json",
        "v24_protocol": repo / "results/online/proximal_v24/protocol.json",
        "v24_final": repo / "results/online/proximal_v24/final/final_test.json",
    }
    actual = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "v23_protocol": V23_PROTOCOL_SHA256,
        "v23_final": V23_FINAL_SHA256,
        "v24_protocol": V24_PROTOCOL_SHA256,
        "v24_final": V24_FINAL_SHA256,
    }
    trees = {
        "v23": _git_output(repo, "rev-parse", f"{commit}:results/online/proximal_v23"),
        "v24": _git_output(repo, "rev-parse", f"{commit}:results/online/proximal_v24"),
    }
    checks = {
        "v23_protocol_byte_unchanged": actual["v23_protocol"]
        == expected["v23_protocol"],
        "v23_final_byte_unchanged": actual["v23_final"] == expected["v23_final"],
        "v23_result_tree_unchanged": trees["v23"] == V23_RESULT_GIT_TREE,
        "v24_protocol_byte_unchanged": actual["v24_protocol"]
        == expected["v24_protocol"],
        "v24_final_byte_unchanged": actual["v24_final"] == expected["v24_final"],
        "v24_result_tree_unchanged": trees["v24"] == V24_RESULT_GIT_TREE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v23/v24 immutable boundary changed: {checks}")
    return {
        "unchanged": True,
        "checks": checks,
        "sha256": actual,
        "git_trees": trees,
        "rerun_or_recomputed": False,
    }


def _candidate_grid() -> list[dict[str, Any]]:
    return [
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output.resolve()
    supersedes = None if args.supersedes is None else args.supersedes.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before freezing v25")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v25 base checkpoint differs from frozen pi0")
    result_root = repo / "results/online/proximal_v25"
    supersession = None
    if result_root.exists():
        if supersedes is None or not supersedes.is_file():
            raise RuntimeError(
                "existing v25 evidence requires an explicit zero-episode protocol "
                "to supersede"
            )
        relative_supersedes = supersedes.relative_to(repo)
        existing_files = {
            path.relative_to(repo) for path in result_root.rglob("*") if path.is_file()
        }
        if existing_files != {relative_supersedes}:
            raise RuntimeError(
                "unexpected v25 evidence exists before revision-2 pre-calibration freeze: "
                f"{sorted(map(str, existing_files))}"
            )
        prior = json.loads(supersedes.read_text())
        if (
            prior.get("protocol_id") != PROTOCOL_ID
            or prior.get("prospective_execution", {}).get(
                "v25_simulator_episode_started"
            )
            is not False
        ):
            raise RuntimeError("superseded protocol does not prove zero v25 episodes")
        committed_prior = subprocess.run(
            ["git", "show", f"{commit}:{relative_supersedes}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if committed_prior != supersedes.read_bytes():
            raise RuntimeError("superseded v25 protocol is not committed at HEAD")
        supersession = {
            "revision": 2,
            "supersedes_file": str(relative_supersedes),
            "supersedes_sha256": file_sha256(supersedes),
            "superseded_before_any_v25_simulator_episode": True,
            "reason": (
                "pre-execution audit unified calibration/adaptation/final evaluation "
                "on the fixed deployment environment, keyed kick debouncing by "
                "toe/riser identity, and pooled interventions per crossed riser"
            ),
            "outcomes_observed_before_revision": False,
        }
    elif supersedes is not None:
        raise RuntimeError("--supersedes was provided but no prior v25 result exists")
    source_hashes = _verify_committed_sources(repo, commit)
    prior_audit = _prior_immutable_audit(repo, commit)
    randomness = fresh_randomness_report(repo)
    if not randomness["passed"]:
        raise RuntimeError(
            f"v25 fresh randomness collision: {randomness['collisions']}"
        )

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": "prospectively_frozen_before_v25_base_only_paired_calibration",
        "revision": 2 if supersession is not None else 1,
        "supersession": supersession,
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
            "all_execution_sources_committed_before_first_v25_episode": True,
        },
        "prior_results_immutable": prior_audit,
        "base_checkpoint": {
            "reference": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "environment": fixed_environment_parameters(),
        "shift_family": {
            "context_id": CONTEXT_ID,
            "family": CONTEXT_FAMILY,
            "fixed_within_and_across_episodes_after_selection": True,
            "only_changed_axis": (
                "current swing leg hip-pitch/knee/ankle-pitch raw-action gain"
            ),
            "stance_leg_gain": 1.0,
            "other_joint_gain": 1.0,
            "geometry_friction_command_controller_nominal": True,
            "actor_observation_interface_unchanged": True,
            "cbf_uses_exact_generated_riser_geometry": True,
            "candidate_grid": _candidate_grid(),
        },
        "calibration": {
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
        },
        "training": formal_algorithm_parameters(),
        "learning_semantics": {
            "one_actor_original_405D_observation": True,
            "one_privileged_critic_original_838D_observation": True,
            "runtime_cbf_executes_filtered_action": True,
            "ppo_ratio_uses_sampled_raw_policy_action": True,
            "teacher_target": (
                "stop-gradient inverse-plant actor-coordinate CBF safe action"
            ),
            "teacher_reprojection_audited_at_1e-6": True,
            "teacher_requires_actual_intervention": True,
            "teacher_requires_next_riser_within_H": True,
            "teacher_requires_no_fall_within_H": True,
            "teacher_weight_clipped_by_actor_coordinate_correction": True,
            "moving_reference": "current round-start pi_k",
            "final_policy": "unconditional round-8 actor",
        },
        "evaluation": {
            "conditions": ["pi0_off", "pi0_on", "pi8_on", "pi8_off"],
            "fresh_paired_conditions": FINAL_EPISODES,
            "batch_size": EVAL_BATCH_SIZE,
            "seed_base": FINAL_SEED_BASE,
            "all_four_conditions_share_initial_state_signatures": True,
            "deterministic_policy_mean": True,
        },
        "excluded": {
            "failure_bank": True,
            "state_restart": True,
            "candidate_line_search": True,
            "performance_gate_or_best_checkpoint": True,
            "new_actor_observation": True,
            "multiple_adaptation_seeds": True,
            "random_per_riser_geometry": True,
            "hidden_variable_shift": True,
        },
        "randomness_preflight": randomness,
        "prospective_execution": {
            "v25_simulator_episode_started": False,
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
