"""Select the first qualifying pure low-friction context with pi0 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from proximal_v24_protocol import (
    BASE_CHECKPOINT_SHA256,
    CALIBRATION_CANDIDATE_PARAMETER_SEEDS,
    CALIBRATION_EPISODES,
    CALIBRATION_FRICTIONS,
    CALIBRATION_MINIMUM_FALLS,
    CALIBRATION_MINIMUM_PURITY,
    CALIBRATION_REPEATS,
    CALIBRATION_SUCCESS_BOUNDS,
    CONTEXT_ID,
    CONTEXT_MODE,
    EXPERIMENT_NAME,
    POLICY_METHOD,
    PROTOCOL_ID,
    TARGET_FAILURE_TYPE,
    calibration_evaluation_seed,
    calibration_gate,
    pure_contact_context_audit,
    validate_v24_calibrated_context,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--precalibration-protocol", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if immutable and path.exists() and path.read_text() != rendered:
        raise RuntimeError(f"refusing to overwrite a different v24 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


def _validate_precalibration_boundary(
    repo: Path,
    protocol_path: Path,
    *,
    protocol_commit: str,
    checkpoint: Path,
) -> dict[str, Any]:
    current_commit = _git_output(repo, "rev-parse", "HEAD")
    if current_commit != protocol_commit:
        raise RuntimeError("v24 calibration HEAD differs from its protocol commit")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("v24 calibration requires a clean tracked worktree")
    relative = protocol_path.relative_to(repo)
    committed = subprocess.run(
        ["git", "show", f"{current_commit}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != file_sha256(protocol_path):
        raise RuntimeError("v24 pre-calibration protocol differs from its Git blob")
    protocol = json.loads(protocol_path.read_text())
    implementation_commit = protocol.get("implementation_boundary", {}).get(
        "git_commit"
    )
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(implementation_commit), "HEAD"],
            cwd=repo,
            check=False,
        ).returncode
        == 0
    )
    calibration = protocol.get("calibration", {})
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "status": protocol.get("status")
        == "prospectively_frozen_before_v24_base_only_calibration",
        "implementation_is_ancestor": ancestor,
        "base_checkpoint": protocol.get("base_checkpoint", {}).get("sha256")
        == file_sha256(checkpoint)
        == BASE_CHECKPOINT_SHA256,
        "randomness": protocol.get("randomness_preflight", {}).get("passed") is True,
        "base_only": calibration.get("base_policy_only") is True,
        "adapted_absent": calibration.get("adapted_policy_evaluations_used") is False,
        "first_qualifier": calibration.get("first_qualifying_candidate_is_frozen")
        is True,
        "candidate_parameter_seeds": calibration.get("candidate_parameter_seeds")
        == list(CALIBRATION_CANDIDATE_PARAMETER_SEEDS),
        "candidate_frictions": calibration.get("candidate_foot_frictions")
        == list(CALIBRATION_FRICTIONS),
        "episodes": calibration.get("episodes_per_candidate") == CALIBRATION_EPISODES,
        "minimum_falls": calibration.get("minimum_fall_count")
        == CALIBRATION_MINIMUM_FALLS,
        "minimum_purity": calibration.get("minimum_contact_purity")
        == CALIBRATION_MINIMUM_PURITY,
        "bounds": calibration.get("success_rate_bounds_inclusive")
        == list(CALIBRATION_SUCCESS_BOUNDS),
        "execution_not_started": protocol.get("prospective_execution", {}).get(
            "calibration_started"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v24 pre-calibration protocol mismatch: {checks}")
    return {
        "file": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "git_commit": current_commit,
        "implementation_commit": implementation_commit,
        "validation": checks,
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_candidate(
    *,
    repo: Path,
    checkpoint: Path,
    context_path: Path,
    candidate_dir: Path,
    candidate_index: int,
    device: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    for repeat in range(CALIBRATION_REPEATS):
        seed = calibration_evaluation_seed(candidate_index, repeat)
        stem = f"DQHMED-seed{seed}"
        output_json = candidate_dir / "evaluation" / f"{stem}.json"
        output_csv = candidate_dir / "evaluation" / f"{stem}.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            try:
                candidate = json.loads(output_json.read_text())
            except (json.JSONDecodeError, OSError):
                candidate = None
            if (
                isinstance(candidate, dict)
                and candidate.get("seed") == seed
                and candidate.get("num_episodes") == 128
                and candidate.get("task") == "Unitree-G1-Stairs-Online-DQHMED"
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_proximal_v24.py"),
                "--repo",
                str(repo),
                "--task",
                "Unitree-G1-Stairs-Online-DQHMED",
                "--checkpoint",
                str(checkpoint),
                "--num-envs",
                "128",
                "--num-episodes",
                "128",
                "--seed",
                str(seed),
                "--device",
                device,
                "--deployment-context",
                str(context_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            completed = subprocess.run(
                command, cwd=repo, check=False, capture_output=True, text=True
            )
            if completed.returncode != 0:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"isolated v24 calibration evaluation failed for {stem}:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        summaries.append(summary)
        rows.extend(_load_rows(output_csv))
    actor_hashes = {summary["actor_state_sha256"] for summary in summaries}
    if len(actor_hashes) != 1:
        raise RuntimeError("v24 calibration batches did not use one frozen pi0 actor")
    return summaries, rows


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint = args.base_checkpoint.resolve()
    protocol_path = args.precalibration_protocol.resolve()
    output_dir = args.output_dir.resolve()
    context_output = args.context_output.resolve()
    for path in (checkpoint, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_reference = _validate_precalibration_boundary(
        repo,
        protocol_path,
        protocol_commit=args.protocol_commit,
        checkpoint=checkpoint,
    )
    marker = output_dir / "calibration_started.json"
    if marker.exists() and not args.resume:
        raise RuntimeError("v24 base-only calibration has already been started")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        marker,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "precalibration_protocol": protocol_reference,
            "base_policy_only": True,
            "adapted_policy_outcomes_observed": False,
            "candidate_attempt_count_before_start": 0,
        },
        immutable=True,
    )

    sys.path.insert(0, str(repo))
    import mjlab.tasks  # noqa: F401

    import src.tasks  # noqa: F401
    from src.tasks.stairs_cbf.deployment_context import (
        generate_v22_specialist_context,
        validate_frozen_deployment_context,
    )

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate_index, parameter_seed in enumerate(
        CALIBRATION_CANDIDATE_PARAMETER_SEEDS
    ):
        candidate = generate_v22_specialist_context(CONTEXT_ID, parameter_seed)
        candidate = validate_frozen_deployment_context(candidate)
        pure_audit = pure_contact_context_audit(
            candidate, candidate_index=candidate_index
        )
        candidate_dir = (
            output_dir
            / f"candidate_{candidate_index:02d}_mu{CALIBRATION_FRICTIONS[candidate_index]:.6f}"
        )
        candidate_path = candidate_dir / "context.json"
        _write_json(candidate_path, candidate, immutable=True)
        summaries, rows = _run_candidate(
            repo=repo,
            checkpoint=checkpoint,
            context_path=candidate_path,
            candidate_dir=candidate_dir,
            candidate_index=candidate_index,
            device=args.device,
            resume=args.resume,
        )
        if len(rows) != CALIBRATION_EPISODES:
            raise RuntimeError("v24 calibration candidate does not have 512 rows")
        success_count = sum(row["success"].lower() == "true" for row in rows)
        fall_count = sum(row["fell"].lower() == "true" for row in rows)
        timeout_count = sum(row["timed_out"].lower() == "true" for row in rows)
        contact_fall_count = sum(
            row["fell"].lower() == "true" and row["failure_type"] == TARGET_FAILURE_TYPE
            for row in rows
        )
        non_success_count = CALIBRATION_EPISODES - success_count
        gate = calibration_gate(
            success_count=success_count,
            fall_count=fall_count,
            contact_fall_count=contact_fall_count,
            non_success_count=non_success_count,
        )
        failure_type_counts: dict[str, int] = {}
        for row in rows:
            if row["fell"].lower() == "true":
                failure_type = row["failure_type"]
                failure_type_counts[failure_type] = (
                    failure_type_counts.get(failure_type, 0) + 1
                )
        attempt = {
            "candidate_index": candidate_index,
            "candidate_parameter_seed": parameter_seed,
            "candidate_foot_friction": CALIBRATION_FRICTIONS[candidate_index],
            "parameters_sha256": candidate["parameters_sha256"],
            "context_file_sha256": file_sha256(candidate_path),
            "pure_contact_context_audit": pure_audit,
            "base_policy_only": True,
            "actor_state_sha256": summaries[0]["actor_state_sha256"],
            "evaluation_seeds": [
                calibration_evaluation_seed(candidate_index, repeat)
                for repeat in range(CALIBRATION_REPEATS)
            ],
            "num_episodes": CALIBRATION_EPISODES,
            "success_count": success_count,
            "success_rate": success_count / CALIBRATION_EPISODES,
            "fall_count": fall_count,
            "fall_rate": fall_count / CALIBRATION_EPISODES,
            "timeout_count": timeout_count,
            "non_success_count": non_success_count,
            "contact_fall_count": contact_fall_count,
            "failure_type_counts": failure_type_counts,
            "contact_purity_over_falls": gate["contact_purity_over_falls"],
            "contact_purity_over_all_non_success": gate[
                "contact_purity_over_all_non_success"
            ],
            "qualification_conditions": gate["conditions"],
            "qualifies": gate["qualifies"],
            "aggregate_telemetry": {
                key: sum(float(summary[key]) for summary in summaries) / len(summaries)
                for key in (
                    "mean_return",
                    "mean_reached_riser",
                    "mean_slip_signal",
                    "mean_contact_mismatch",
                    "intervention_per_riser",
                    "mean_correction_norm",
                    "recovery_takeover_rate",
                )
            },
        }
        attempts.append(attempt)
        progress = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "experiment_name": EXPERIMENT_NAME,
            "policy_method": POLICY_METHOD,
            "context_id": CONTEXT_ID,
            "mode": CONTEXT_MODE,
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "precalibration_protocol": protocol_reference,
            "attempts": attempts,
        }
        _write_json(output_dir / "calibration_progress.json", progress)
        print(json.dumps(attempt, indent=2, sort_keys=True), flush=True)
        if gate["qualifies"]:
            selected = candidate
            break

    if selected is None:
        negative = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "calibration_negative_no_candidate_qualified",
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "all_declared_candidates_evaluated": len(attempts)
            == len(CALIBRATION_CANDIDATE_PARAMETER_SEEDS),
            "attempts": attempts,
            "adaptation_started": False,
            "final_test_started": False,
        }
        _write_json(output_dir / "calibration_negative.json", negative, immutable=True)
        raise RuntimeError("no declared v24 pure-contact candidate passed every gate")

    selected_attempt = attempts[-1]
    selected["calibration"] = {
        "kind": "base_policy_pure_contact_first_qualifying_v24",
        "protocol_id": PROTOCOL_ID,
        "selection_rule": "first base-only pure-friction candidate satisfying every frozen gate",
        "success_rate_bounds_inclusive": list(CALIBRATION_SUCCESS_BOUNDS),
        "minimum_fall_count": CALIBRATION_MINIMUM_FALLS,
        "minimum_contact_purity": CALIBRATION_MINIMUM_PURITY,
        "purity_primary_denominator": "all non-success episodes",
        "purity_falls_only_also_required": True,
        "episodes_per_candidate": CALIBRATION_EPISODES,
        "candidate_parameter_seeds": list(CALIBRATION_CANDIDATE_PARAMETER_SEEDS),
        "candidate_foot_frictions": list(CALIBRATION_FRICTIONS),
        "attempts": attempts,
        "selected_candidate_parameter_seed": selected_attempt[
            "candidate_parameter_seed"
        ],
        "selected_foot_friction": selected_attempt["candidate_foot_friction"],
        "selected_parameters_sha256": selected["parameters_sha256"],
        "base_policy_checkpoint_sha256": file_sha256(checkpoint),
        "adapted_policy_evaluations_used": False,
        "precalibration_protocol_sha256": protocol_reference["sha256"],
        "precalibration_protocol_git_commit": protocol_reference["git_commit"],
    }
    selected = validate_v24_calibrated_context(selected)
    _write_json(context_output, selected, immutable=True)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "first_qualifying_pure_contact_context_frozen",
        "context_id": CONTEXT_ID,
        "mode": CONTEXT_MODE,
        "frozen_context": str(context_output),
        "frozen_context_file_sha256": file_sha256(context_output),
        "parameters_sha256": selected["parameters_sha256"],
        "selected_candidate_parameter_seed": selected_attempt[
            "candidate_parameter_seed"
        ],
        "selected_foot_friction": selected_attempt["candidate_foot_friction"],
        "selected_attempt": selected_attempt,
        "candidate_count_evaluated": len(attempts),
        "base_policy_only": True,
        "adapted_policy_evaluations_used": False,
        "precalibration_protocol": protocol_reference,
    }
    _write_json(output_dir / "calibration_summary.json", summary, immutable=True)
    _write_json(
        output_dir / "calibration_completed.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "context_file_sha256": file_sha256(context_output),
            "parameters_sha256": selected["parameters_sha256"],
            "adaptation_started": False,
            "adapted_policy_outcomes_observed": False,
        },
        immutable=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
