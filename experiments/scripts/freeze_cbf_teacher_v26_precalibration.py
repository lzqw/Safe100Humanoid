"""Freeze v26 implementation and ordered grid before any simulator episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v26_protocol import (
    ALIGNMENT_COVERAGE_MINIMUM,
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CALIBRATION_EPISODES,
    CALIBRATION_HEIGHTS_M,
    CALIBRATION_OFF_SUCCESS_BOUNDS,
    CALIBRATION_ON_SUCCESS_BOUNDS,
    CALIBRATION_REPEATS,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    DEVELOPMENT_SEEDS,
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
            raise RuntimeError(f"v26 source is not committed at {commit}: {relative}")
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
            "riser_height_m": height,
            "severity_order": "light_to_severe",
            "evaluation_seeds": [
                calibration_evaluation_seed(index, repeat)
                for repeat in range(CALIBRATION_REPEATS)
            ],
            "paired_filter_arms_share_each_seed": True,
        }
        for index, height in enumerate(CALIBRATION_HEIGHTS_M)
    ]


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise RuntimeError(f"refusing to overwrite another v26 protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _committed_protocol_chain(
    repo: Path, commit: str, latest: Path, result_root: Path
) -> list[dict[str, Any]]:
    """Validate and return the complete newest-to-oldest supersession chain."""
    chain = []
    seen: set[Path] = set()
    current = latest.resolve()
    while True:
        try:
            relative_to_result = current.relative_to(result_root)
            relative_to_repo = current.relative_to(repo)
        except ValueError as error:
            raise RuntimeError(
                f"superseded protocol is outside the v26 result root: {current}"
            ) from error
        if current in seen:
            raise RuntimeError("v26 supersession chain contains a cycle")
        seen.add(current)
        if not current.is_file():
            raise FileNotFoundError(current)
        content = current.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative_to_repo}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if committed != content:
            raise RuntimeError(
                f"superseded v26 protocol is not committed at HEAD: {relative_to_repo}"
            )
        payload = json.loads(content)
        declared_revision = payload.get("revision")
        legacy_revision_one = (
            declared_revision is None
            and relative_to_result == Path("precalibration_protocol.json")
            and payload.get("supersession") is None
        )
        revision = 1 if legacy_revision_one else declared_revision
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("status")
            != "prospectively_frozen_before_v26_base_only_paired_calibration"
            or not isinstance(revision, int)
            or revision < 1
            or payload.get("prospective_execution", {}).get(
                "v26_simulator_episode_started"
            )
            is not False
        ):
            raise RuntimeError(f"invalid zero-episode v26 protocol: {relative_to_repo}")
        chain.append(
            {
                "revision": revision,
                "file": str(relative_to_repo),
                "result_relative_file": str(relative_to_result),
                "sha256": hashlib.sha256(content).hexdigest(),
                "payload": payload,
            }
        )
        link = payload.get("supersession")
        if link is None:
            break
        next_file = link.get("supersedes_file")
        next_sha = link.get("supersedes_sha256")
        if not isinstance(next_file, str) or not isinstance(next_sha, str):
            raise TypeError("v26 supersession link is incomplete")
        next_path = (repo / next_file).resolve()
        if not next_path.is_file() or file_sha256(next_path) != next_sha:
            raise RuntimeError("v26 supersession link hash does not match its parent")
        current = next_path
    revisions = [item["revision"] for item in chain]
    if revisions != list(range(revisions[0], 0, -1)):
        raise RuntimeError(f"v26 revision chain is not contiguous: {revisions}")
    return chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes", type=Path)
    parser.add_argument("--supersession-reason")
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    output = args.output.resolve()
    supersedes = None if args.supersedes is None else args.supersedes.resolve()
    supersession_reason = args.supersession_reason
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("worktree must be clean before freezing v26")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v26 base checkpoint differs from frozen pi0")
    result_root = repo / "results/online/proximal_v26"
    supersession = None
    revision = 1
    if result_root.exists():
        if supersedes is None or not supersedes.is_file():
            raise RuntimeError(
                "existing v26 evidence requires an explicit zero-episode protocol "
                "to supersede"
            )
        if not supersession_reason or not supersession_reason.strip():
            raise RuntimeError("a non-empty supersession reason is required")
        relative_supersedes = supersedes.relative_to(repo)
        existing_files = {
            path.relative_to(repo) for path in result_root.rglob("*") if path.is_file()
        }
        chain = _committed_protocol_chain(repo, commit, supersedes, result_root)
        expected_files = {Path(item["file"]) for item in chain}
        if existing_files != expected_files:
            raise RuntimeError(
                "unexpected v26 evidence exists before successor pre-calibration freeze: "
                f"{sorted(map(str, existing_files))}"
            )
        prior = chain[0]
        revision = int(prior["revision"]) + 1
        supersession = {
            "revision": revision,
            "supersedes_revision": prior["revision"],
            "supersedes_file": str(relative_supersedes),
            "supersedes_sha256": prior["sha256"],
            "superseded_before_any_v26_simulator_episode": True,
            "reason": supersession_reason.strip(),
            "outcomes_observed_before_revision": False,
            "verified_protocol_history": [
                {
                    "revision": item["revision"],
                    "file": item["file"],
                    "sha256": item["sha256"],
                }
                for item in chain
            ],
        }
    elif supersedes is not None or supersession_reason is not None:
        raise RuntimeError(
            "--supersedes/--supersession-reason were provided but no prior v26 result exists"
        )
    source_hashes = _verify_committed_sources(repo, commit)
    prior_audit = _prior_immutable_audit(repo, commit)
    randomness = fresh_randomness_report(repo)
    if not randomness["passed"]:
        raise RuntimeError(
            f"v26 fresh randomness collision: {randomness['collisions']}"
        )

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": EXPERIMENT_NAME,
        "policy_method": POLICY_METHOD,
        "status": "prospectively_frozen_before_v26_base_only_paired_calibration",
        "revision": revision,
        "supersession": supersession,
        "prospective_freeze": {
            "before_any_formal_v26_simulator_episode": True,
            "development_episodes_already_observed_and_excluded": True,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "formal_seed_overlap": False,
        },
        "implementation_boundary": {
            "git_commit": commit,
            "source_files": source_hashes,
            "all_execution_sources_committed_before_first_v26_episode": True,
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
            "only_changed_axis": "uniform stair riser height",
            "baseline_riser_height_m": 0.13,
            "clearance_barrier": "sloped_xz",
            "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
            "plant_action_transform": "identity",
            "friction_command_controller_nominal": True,
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
                "debounced entry of the selected swing toe into exact riser overlap"
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
                "stop-gradient actor-coordinate CBF safe action"
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
            "v26_simulator_episode_started": False,
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
