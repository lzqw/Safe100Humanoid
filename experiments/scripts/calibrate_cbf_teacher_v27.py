"""Run the fixed, paired v27 confirmatory calibration."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v27_protocol import (
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_EPISODES,
    CALIBRATION_REPEATS,
    CLEARANCE_BARRIER_SLOPE,
    ENVIRONMENT_VARIANT,
    EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    RECOVERY_DISTANCE_M,
    RISER_HEIGHT_M,
    SOURCE_FILES,
    TASK_ID,
    calibration_gate,
    calibration_seed,
    canonical_sha256,
    file_sha256,
    fixed_environment_parameters,
)


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
    runtime_filter: str,
    seed: int,
    output_dir: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    arm = output_dir / f"seed_{seed}" / runtime_filter
    output_json = arm / "summary.json"
    output_csv = arm / "episodes.csv"
    summary = None
    if resume and output_json.is_file() and output_csv.is_file():
        candidate = json.loads(output_json.read_text())
        if (
            candidate.get("seed") == seed
            and candidate.get("runtime_filter") == (runtime_filter == "on")
            and candidate.get("riser_height_m") == RISER_HEIGHT_M
            and candidate.get("clearance_barrier_slope") == CLEARANCE_BARRIER_SLOPE
            and candidate.get("recovery_distance_m") == RECOVERY_DISTANCE_M
            and candidate.get("checkpoint_sha256") == file_sha256(checkpoint)
        ):
            summary = candidate
    if summary is None:
        command = [
            sys.executable,
            str(repo / "experiments/scripts/evaluate_cbf_teacher_v26.py"),
            "--repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--riser-height",
            str(RISER_HEIGHT_M),
            "--clearance-slope",
            str(CLEARANCE_BARRIER_SLOPE),
            "--recovery-distance",
            str(RECOVERY_DISTANCE_M),
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
                (completed.stdout + "\n" + completed.stderr).splitlines()[-120:]
            )
            raise RuntimeError(
                f"v27 calibration arm failed ({runtime_filter=}, {seed=}):\n{diagnostic}"
            )
        summary = json.loads(output_json.read_text())
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EVAL_BATCH_SIZE:
        raise RuntimeError("v27 calibration arm has an incomplete episode table")
    return summary, rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    precalibration_path = args.precalibration_protocol.resolve()
    output_dir = args.output_dir.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout
    if status:
        raise RuntimeError("v27 calibration requires a clean committed worktree")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v27 calibration checkpoint differs from frozen pi0")
    precalibration = json.loads(precalibration_path.read_text())
    if (
        precalibration.get("protocol_id") != PROTOCOL_ID
        or precalibration.get("status")
        != "prospectively_frozen_before_v27_confirmatory_calibration"
        or precalibration.get("environment") != fixed_environment_parameters()
        or precalibration.get("freshness", {}).get(
            "before_any_formal_v27_simulator_episode"
        )
        is not True
    ):
        raise RuntimeError("v27 pre-calibration protocol is invalid")
    relative_protocol = precalibration_path.relative_to(repo)
    committed_protocol = subprocess.run(
        ["git", "show", f"HEAD:{relative_protocol}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_protocol != precalibration_path.read_bytes():
        raise RuntimeError("v27 pre-calibration protocol is not committed")
    implementation = precalibration["implementation_boundary"]
    implementation_commit = implementation["git_commit"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=repo,
    ).returncode:
        raise RuntimeError("v27 implementation commit is not an ancestor of HEAD")
    if set(implementation["source_files"]) != set(SOURCE_FILES) or any(
        file_sha256(repo / relative) != implementation["source_files"][relative]
        for relative in SOURCE_FILES
    ):
        raise RuntimeError("v27 source differs from the prospective freeze")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = output_dir / "calibration_execution_started.json"
    if started.exists() and not args.resume:
        raise RuntimeError("v27 calibration already started; use --resume")
    if not started.exists():
        _atomic_json(
            started,
            {
                "protocol_id": PROTOCOL_ID,
                "precalibration_protocol_sha256": file_sha256(precalibration_path),
                "base_policy_only": True,
                "single_fixed_candidate": True,
            },
        )

    off_summaries = []
    on_summaries = []
    off_rows: list[dict[str, str]] = []
    on_rows: list[dict[str, str]] = []
    for repeat in range(CALIBRATION_REPEATS):
        seed = calibration_seed(repeat)
        off_summary, off_batch = _run_arm(
            repo=repo,
            checkpoint=checkpoint,
            runtime_filter="off",
            seed=seed,
            output_dir=output_dir / "candidate",
            device=args.device,
            resume=args.resume,
        )
        on_summary, on_batch = _run_arm(
            repo=repo,
            checkpoint=checkpoint,
            runtime_filter="on",
            seed=seed,
            output_dir=output_dir / "candidate",
            device=args.device,
            resume=args.resume,
        )
        if off_summary["initial_state_signature"] != on_summary["initial_state_signature"]:
            raise RuntimeError("v27 off/on initial conditions differ")
        if off_summary["actor_state_sha256"] != on_summary["actor_state_sha256"]:
            raise RuntimeError("v27 off/on arms do not share pi0")
        off_summaries.append(off_summary)
        on_summaries.append(on_summary)
        off_rows.extend(off_batch)
        on_rows.extend(on_batch)

    def identity(row: dict[str, str]) -> tuple[int, int]:
        return int(row["evaluation_seed"]), int(row["environment_id"])

    off_rows.sort(key=identity)
    on_rows.sort(key=identity)
    if len(off_rows) != CALIBRATION_EPISODES or len(on_rows) != CALIBRATION_EPISODES:
        raise RuntimeError("v27 confirmatory episode count is incomplete")
    if [identity(row) for row in off_rows] != [identity(row) for row in on_rows]:
        raise RuntimeError("v27 paired identities differ")
    actor_hashes = {
        item["actor_state_sha256"] for item in (*off_summaries, *on_summaries)
    }
    if len(actor_hashes) != 1:
        raise RuntimeError("v27 actor changed during calibration")

    off_success = [_bool(row["success"]) for row in off_rows]
    on_success = [_bool(row["success"]) for row in on_rows]
    off_kick = [_bool(row["toe_riser_kick"]) for row in off_rows]
    failures = sum(not value for value in off_success)
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )
    rescued = sum(
        (not off) and on
        for off, on in zip(off_success, on_success, strict=True)
    )
    regressed = sum(
        off and (not on)
        for off, on in zip(off_success, on_success, strict=True)
    )
    gate = calibration_gate(
        off_success_count=sum(off_success),
        on_success_count=sum(on_success),
        off_toe_riser_failure_count=aligned,
        off_failure_count=failures,
        rescued_count=rescued,
    )
    attempt = {
        "candidate_index": 0,
        "riser_height_m": RISER_HEIGHT_M,
        "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
        "recovery_distance_m": RECOVERY_DISTANCE_M,
        "evaluation_seeds": [calibration_seed(i) for i in range(CALIBRATION_REPEATS)],
        "actor_state_sha256": actor_hashes.pop(),
        "off_initial_state_signatures": [
            item["initial_state_signature"] for item in off_summaries
        ],
        "on_initial_state_signatures": [
            item["initial_state_signature"] for item in on_summaries
        ],
        "regressed_count": regressed,
        **gate,
    }
    _atomic_json(output_dir / "attempt.json", attempt)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "qualified_for_training" if gate["qualifies"] else "did_not_qualify",
        "base_policy_only": True,
        "adapted_policy_evaluations_used": False,
        "single_fixed_candidate": True,
        "attempt": attempt,
        "precalibration_protocol_sha256": file_sha256(precalibration_path),
    }
    _atomic_json(output_dir / "calibration_summary.json", summary)
    if gate["qualifies"]:
        parameters = {
            **fixed_environment_parameters(),
            "candidate_index": 0,
        }
        context = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "parameters": parameters,
            "parameters_sha256": canonical_sha256(parameters),
            "calibration": attempt,
        }
        _atomic_json(output_dir / "calibrated_context.json", context)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
