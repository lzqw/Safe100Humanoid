"""Run ordered paired base-only rescue calibration for v25."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v25_protocol import (
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_EPISODES,
    CALIBRATION_GAINS,
    CALIBRATION_REPEATS,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    ENVIRONMENT_VARIANT,
    EVAL_BATCH_SIZE,
    PRECALIBRATION_REVISION,
    PROTOCOL_ID,
    SOURCE_FILES,
    TASK_ID,
    calibration_evaluation_seed,
    calibration_gate,
    canonical_sha256,
    fixed_environment_parameters,
)
from proximal_v23_io import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--precalibration-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid CSV boolean: {value!r}")
    return normalized == "true"


def _run_arm(
    *,
    repo: Path,
    checkpoint: Path,
    gain: float,
    runtime_filter: str,
    seed: int,
    output_dir: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm_dir = output_dir / f"seed_{seed}" / runtime_filter
    output_json = arm_dir / "summary.json"
    output_csv = arm_dir / "episodes.csv"
    summary = None
    if resume and output_json.is_file() and output_csv.is_file():
        candidate = json.loads(output_json.read_text())
        if (
            candidate.get("seed") == seed
            and candidate.get("task") == TASK_ID
            and candidate.get("environment_variant") == ENVIRONMENT_VARIANT
            and candidate.get("num_envs") == EVAL_BATCH_SIZE
            and candidate.get("num_episodes") == EVAL_BATCH_SIZE
            and candidate.get("runtime_filter") == (runtime_filter == "on")
            and candidate.get("swing_underresponse_gain") == gain
            and candidate.get("checkpoint_sha256") == file_sha256(checkpoint)
        ):
            summary = candidate
    if summary is None:
        command = [
            sys.executable,
            str(repo / "experiments/scripts/evaluate_cbf_teacher_v25.py"),
            "--repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--gain",
            str(gain),
            "--runtime-filter",
            runtime_filter,
            "--num-envs",
            str(EVAL_BATCH_SIZE),
            "--num-episodes",
            str(EVAL_BATCH_SIZE),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
        completed = subprocess.run(
            command, cwd=repo, check=False, capture_output=True, text=True
        )
        if completed.returncode:
            diagnostic = "\n".join(
                (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
            )
            raise RuntimeError(
                f"v25 calibration arm failed ({gain=}, {runtime_filter=}):\n{diagnostic}"
            )
        summary = json.loads(output_json.read_text())
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EVAL_BATCH_SIZE:
        raise RuntimeError("v25 calibration arm has an incomplete episode table")
    return summary, rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    precalibration_path = args.precalibration_protocol.resolve()
    output_dir = args.output_dir.resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("v25 calibration requires a clean committed worktree")
    if not checkpoint.is_file() or not precalibration_path.is_file():
        raise FileNotFoundError("base checkpoint or pre-calibration protocol missing")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v25 calibration checkpoint differs from frozen pi0")
    precalibration = json.loads(precalibration_path.read_text())
    if (
        precalibration.get("protocol_id") != PROTOCOL_ID
        or precalibration.get("status")
        != "prospectively_frozen_before_v25_base_only_paired_calibration"
    ):
        raise RuntimeError("v25 pre-calibration protocol is not the frozen input")
    if (
        precalibration.get("revision") != PRECALIBRATION_REVISION
        or precalibration.get("environment") != fixed_environment_parameters()
        or precalibration.get("supersession", {}).get(
            "superseded_before_any_v25_simulator_episode"
        )
        is not True
    ):
        raise RuntimeError(
            f"v25 revision-{PRECALIBRATION_REVISION} fixed-environment boundary "
            "is missing"
        )
    implementation = precalibration.get("implementation_boundary", {})
    implementation_commit = str(implementation.get("git_commit", ""))
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode:
        raise RuntimeError("v25 frozen implementation is not an ancestor of HEAD")
    bound_sources = implementation.get("source_files", {})
    if set(bound_sources) != set(SOURCE_FILES) or any(
        file_sha256(repo / relative) != bound_sources.get(relative)
        for relative in SOURCE_FILES
    ):
        raise RuntimeError("v25 calibration source differs from pre-calibration freeze")
    relative_precalibration = precalibration_path.relative_to(repo)
    committed_precalibration = subprocess.run(
        ["git", "show", f"HEAD:{relative_precalibration}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_precalibration != precalibration_path.read_bytes():
        raise RuntimeError("v25 pre-calibration protocol is not committed at HEAD")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = output_dir / "calibration_execution_started.json"
    if started.exists() and not args.resume:
        raise RuntimeError("v25 calibration has already started; use --resume")
    if not started.exists():
        _atomic_json(
            started,
            {
                "protocol_id": PROTOCOL_ID,
                "precalibration_protocol_sha256": file_sha256(precalibration_path),
                "base_policy_only": True,
                "adapted_policy_evaluations_used": False,
                "ordered_first_qualifier_rule": True,
            },
        )

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_off_rows: list[dict[str, Any]] = []
    selected_on_rows: list[dict[str, Any]] = []
    for candidate_index, gain in enumerate(CALIBRATION_GAINS):
        off_summaries = []
        on_summaries = []
        off_rows: list[dict[str, Any]] = []
        on_rows: list[dict[str, Any]] = []
        for repeat in range(CALIBRATION_REPEATS):
            seed = calibration_evaluation_seed(candidate_index, repeat)
            off_summary, off_batch = _run_arm(
                repo=repo,
                checkpoint=checkpoint,
                gain=gain,
                runtime_filter="off",
                seed=seed,
                output_dir=output_dir
                / "candidates"
                / f"candidate_{candidate_index:02d}",
                device=args.device,
                resume=args.resume,
            )
            on_summary, on_batch = _run_arm(
                repo=repo,
                checkpoint=checkpoint,
                gain=gain,
                runtime_filter="on",
                seed=seed,
                output_dir=output_dir
                / "candidates"
                / f"candidate_{candidate_index:02d}",
                device=args.device,
                resume=args.resume,
            )
            if (
                off_summary["initial_state_signature"]
                != on_summary["initial_state_signature"]
            ):
                raise RuntimeError("v25 off/on arms do not share initial conditions")
            if off_summary["actor_state_sha256"] != on_summary["actor_state_sha256"]:
                raise RuntimeError("v25 off/on arms do not share pi0")
            off_summaries.append(off_summary)
            on_summaries.append(on_summary)
            off_rows.extend(off_batch)
            on_rows.extend(on_batch)

        def identity(row: dict[str, str]) -> tuple[int, int]:
            return int(row["evaluation_seed"]), int(row["environment_id"])

        off_rows.sort(key=identity)
        on_rows.sort(key=identity)
        if [identity(row) for row in off_rows] != [identity(row) for row in on_rows]:
            raise RuntimeError("v25 paired calibration identities differ")
        actor_hashes = {
            summary["actor_state_sha256"] for summary in (*off_summaries, *on_summaries)
        }
        if len(actor_hashes) != 1:
            raise RuntimeError("v25 calibration changed pi0 between paired batches")
        off_success = [_bool(row["success"]) for row in off_rows]
        on_success = [_bool(row["success"]) for row in on_rows]
        off_kick = [_bool(row["toe_riser_kick"]) for row in off_rows]
        off_failure_count = sum(not value for value in off_success)
        aligned_failures = sum(
            (not success) and kick
            for success, kick in zip(off_success, off_kick, strict=True)
        )
        rescued = sum(
            (not off) and on for off, on in zip(off_success, on_success, strict=True)
        )
        gate = calibration_gate(
            off_success_count=sum(off_success),
            on_success_count=sum(on_success),
            off_toe_riser_failure_count=aligned_failures,
            off_failure_count=off_failure_count,
            rescued_count=rescued,
            paired_count=CALIBRATION_EPISODES,
        )
        attempt = {
            "candidate_index": candidate_index,
            "swing_underresponse_gain": gain,
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "evaluation_seeds": [
                calibration_evaluation_seed(candidate_index, repeat)
                for repeat in range(CALIBRATION_REPEATS)
            ],
            "actor_state_sha256": actor_hashes.pop(),
            "off_initial_state_signatures": [
                summary["initial_state_signature"] for summary in off_summaries
            ],
            "on_initial_state_signatures": [
                summary["initial_state_signature"] for summary in on_summaries
            ],
            **gate,
        }
        attempts.append(attempt)
        _atomic_json(output_dir / "attempts.json", attempts)
        print(json.dumps(attempt, sort_keys=True), flush=True)
        if gate["qualifies"]:
            selected = attempt
            selected_off_rows = off_rows
            selected_on_rows = on_rows
            break

    if selected is None:
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "no_candidate_qualified",
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "candidate_count_evaluated": len(attempts),
            "attempts": attempts,
            "precalibration_protocol_sha256": file_sha256(precalibration_path),
        }
        _atomic_json(output_dir / "calibration_summary.json", summary)
        raise RuntimeError("no v25 actuator-gain candidate entered the rescue corridor")

    parameters = {
        "context_id": CONTEXT_ID,
        "family": CONTEXT_FAMILY,
        "selected_candidate_index": selected["candidate_index"],
        "swing_underresponse_gain": selected["swing_underresponse_gain"],
        "phase_selective": True,
        "affected_joint_suffixes": [
            "hip_pitch_joint",
            "knee_joint",
            "ankle_pitch_joint",
        ],
        "stance_leg_gain": 1.0,
        "other_joint_gain": 1.0,
        "environment_variant": ENVIRONMENT_VARIANT,
        "actor_observation_corruption": "disabled",
        "encoder_bias": "absent",
        "curriculum": "disabled",
        "fresh_initial_state_reset_events": ["reset_base", "reset_robot_joints"],
    }
    context = {
        "schema_version": 1,
        "context_id": CONTEXT_ID,
        "parameters_sha256": canonical_sha256(parameters),
        "shift": {
            "family": CONTEXT_FAMILY,
            "selected_candidate_index": selected["candidate_index"],
            "swing_underresponse_gain": selected["swing_underresponse_gain"],
            "phase_selective": True,
            "affected_joint_suffixes": parameters["affected_joint_suffixes"],
            "stance_leg_gain": 1.0,
            "other_joint_gain": 1.0,
            "terrain_geometry": "nominal_fixed_DQHMED",
            "friction": "nominal",
            "command": "nominal",
            "controller": "nominal",
            "observation_interface": "original_405D",
            "cbf_geometry": "exact_generated_riser_metadata",
            "environment_variant": parameters["environment_variant"],
            "actor_observation_corruption": parameters["actor_observation_corruption"],
            "encoder_bias": parameters["encoder_bias"],
            "curriculum": parameters["curriculum"],
            "fresh_initial_state_reset_events": parameters[
                "fresh_initial_state_reset_events"
            ],
        },
        "calibration": {
            "kind": "base_policy_paired_cbf_rescue_first_qualifier_v25",
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "precalibration_protocol_sha256": file_sha256(precalibration_path),
            "attempts": attempts,
        },
    }
    context_path = output_dir / "context.json"
    _atomic_json(context_path, context)

    paired_path = output_dir / "selected_paired_episodes.csv"
    paired_rows = []
    for off, on in zip(selected_off_rows, selected_on_rows, strict=True):
        row = {
            "evaluation_seed": off["evaluation_seed"],
            "environment_id": off["environment_id"],
            "off_success": off["success"],
            "on_success": on["success"],
            "off_fell": off["fell"],
            "on_fell": on["fell"],
            "off_toe_riser_kick": off["toe_riser_kick"],
            "on_toe_riser_kick": on["toe_riser_kick"],
            "off_failure_type": off["failure_type"],
            "on_failure_type": on["failure_type"],
            "off_max_riser": off["max_riser"],
            "on_max_riser": on["max_riser"],
            "off_would_intervene_count": off["would_intervene_count"],
            "on_intervention_count": on["intervention_count"],
            "off_would_intervene_per_riser": off["would_intervene_per_riser"],
            "on_intervention_per_riser": on["intervention_per_riser"],
        }
        paired_rows.append(row)
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "first_qualifying_candidate_frozen",
        "base_policy_only": True,
        "adapted_policy_evaluations_used": False,
        "candidate_count_evaluated": len(attempts),
        "ordered_light_to_severe": True,
        "selected_candidate_index": selected["candidate_index"],
        "selected_swing_underresponse_gain": selected["swing_underresponse_gain"],
        "selected_actor_state_sha256": selected["actor_state_sha256"],
        "selected_gate": selected,
        "parameters_sha256": context["parameters_sha256"],
        "frozen_context_file_sha256": file_sha256(context_path),
        "selected_paired_episode_count": len(paired_rows),
        "selected_paired_csv_sha256": file_sha256(paired_path),
        "precalibration_protocol_sha256": file_sha256(precalibration_path),
        "attempts": attempts,
    }
    _atomic_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
